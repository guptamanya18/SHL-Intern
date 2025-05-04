from dataset_creation import MGSNetDatasetPreparer

from dataset_creation import MGSNetTestDatasetPreparer
# Example usage
dataset_preparer = MGSNetTestDatasetPreparer(
    audio_dir=r'E:\Hackathon\SHL_Intern\assets\Dataset\audios\test',
    transcript_dir=r'E:\Hackathon\SHL_Intern\assets\Dataset\test.csv',
    output_dir=r'E:\Hackathon\SHL_Intern\assets\Dataset\test_data'
)
dataset_preparer.create_dataset()
