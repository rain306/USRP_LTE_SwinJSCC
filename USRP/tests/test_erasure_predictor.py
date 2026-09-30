"""Fast, synthetic CPU checks for the optional predictor branch."""
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from USRP.predictor.erasure_predictor import chunk_features, load_source, sender
from USRP.predictor.evaluate_erasure_predictor import evaluate, receive_symbols
from USRP.core_codec.latent_storage import quantize
from USRP.predictor.score_erasure_predictor import score
from USRP.core_codec.swinjscc_codec import embed_meta_into_bytes
from USRP.predictor.train_erasure_predictor import train


class PredictorTests(unittest.TestCase):
    def make_source(self, directory, index):
        rng = np.random.RandomState(index)
        symbols = (rng.normal(size=6000) + 1j*rng.normal(size=6000)).astype(np.complex64)
        metadata = dict(image_stem='test_{}'.format(index),
                        model='SwinJSCC_w/o_SAandRA', C=96, snr_db=10,
                        input_shape=[1, 3, 512, 768], feature_shape=[1, 125, 96],
                        original_symbol_power=2.0)
        packed, meta = quantize(symbols, metadata)
        stream = io.BytesIO()
        np.save(stream, packed, allow_pickle=False)
        raw = embed_meta_into_bytes(stream.getvalue(), meta)
        path = directory/('test_{}.npy'.format(index))
        path.write_bytes(raw)
        features, eligible = chunk_features(raw, meta, 1200)
        self.assertEqual(len(features), (len(raw)+1199)//1200)
        self.assertFalse(eligible[0])
        energy, _ = sender.energy_scores(raw, 1200, 'int')
        labels = [float(value + index*0.1) if eligible[i] else 0.0
                  for i, value in enumerate(energy)]
        provenance = dict(scheme='single_chunk_erasure_delta_mse_v1',
                          source_sha256=hashlib.sha256(raw).hexdigest(),
                          chunk_bytes=1200, total_chunks=len(labels),
                          storage_format='swinjscc_int8_v1', bytes_per_symbol=2,
                          model=meta['model'], C=96, snr=10,
                          checkpoint_sha256='encoder_test',
                          decoder_checkpoint_sha256='decoder_test')
        path.with_suffix('.erasure.json').write_text(json.dumps(dict(
            provenance=provenance, complete=True, scores=labels)), encoding='utf-8')
        return path

    def test_training_bundle_and_sender(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root/'source'
            source.mkdir()
            for index in range(5):
                self.make_source(source, index)
            model, report = train(source, root/'modelrun', heldout_count=2,
                                  expected_images=5)
            self.assertEqual(report['training_images'], 3)
            self.assertEqual(report['heldout_images'], 2)
            self.assertFalse(set(model['training_image_sha256']) &
                             set(model['heldout_image_sha256']))
            self.assertEqual(score(source, root/'modelrun'/'model.json',
                                   root/'bundle'), 2)
            copied = sorted((root/'bundle').glob('*.npy'))
            self.assertEqual(len(copied), 2)
            raw, _ = load_source(copied[0])
            _, meta = load_source(copied[0])
            values, scheme = sender.predictor_scores(raw, str(copied[0]), 1200)
            self.assertEqual(len(values), (len(raw)+1199)//1200)
            self.assertEqual(scheme, 'int8_erasure_rank_ridge_v1')
            _, _, _, control = sender.plan(raw, 1200, 1, 'semantic',
                                           .10, .10, 2, 1, 'predictor', str(copied[0]))
            self.assertEqual(control['metric'], scheme)
            with self.assertRaises(ValueError):
                sender.predictor_scores(raw+b'x', str(copied[0]), 1200)
            schedule = list(range(len(values)))
            clean_symbols, unique = receive_symbols(raw, meta, 1200, schedule,
                                                     [False]*len(schedule))
            self.assertEqual(unique, len(values))
            self.assertEqual(len(clean_symbols), meta['num_complex_symbols'])
            refs = root/'references'
            refs.mkdir()
            for index in range(5):
                Image.fromarray(np.full((512, 768, 3), 128, dtype=np.uint8)).save(
                    refs/('test_{}.png'.format(index)))

            class FakeCodec:
                def __init__(self, *args, **kwargs):
                    pass

                def decode(self, symbols, feature_shape, input_size, **kwargs):
                    h, w = input_size
                    return np.full((h, w, 3), 127+float(np.mean(symbols.real)))

            def fake_hash(path):
                return 'encoder_test' if Path(path).name == 'encoder.model' else 'decoder_test'

            with patch('USRP.predictor.evaluate_erasure_predictor.SwinJSCCCodec', FakeCodec), \
                    patch('USRP.predictor.evaluate_erasure_predictor.file_hash', fake_hash):
                result = evaluate(root/'bundle', refs, root/'simulation',
                                  root/'encoder.model', root/'decoder.model',
                                  loss_rates=[.1], trials=1, device='cpu')
            self.assertEqual(result['summary'][0]['pairs'], 2)


if __name__ == '__main__':
    unittest.main()
