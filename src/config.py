class Config:
    def __init__(self):
        self.training_rows = 28
        self.training_cols = 28
        self.hand_rows = self.training_cols * 10
        self.hand_cols = self.training_cols * 10
        self.video_width = 640
        self.video_height = 480
        self.ulx = 20
        self.uly = 150 
        self.brx = self.ulx + self.hand_cols
        self.bry = self.uly + self.hand_rows
        self.enabled_windows = ('video', 'Letter Prediction', 'Word Suggestions',
                                'Hand Gesture', 'Skeleton on hand')
