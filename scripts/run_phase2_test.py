from src.pipeline.batch_prediction_pipeline import BatchPredictionPipeline
import traceback

def main():
    print('PHASE2_TEST_START')
    try:
        pipeline = BatchPredictionPipeline()
        pipeline.run_pipeline()
        print('PHASE2_TEST_SUCCESS')
    except Exception as e:
        traceback.print_exc()
        print('PHASE2_TEST_FAILED', e)

if __name__ == '__main__':
    main()
