class Score:
    def __init__(self):
        self.score = 0
        self.nums = 0

    def update(self, score):
        print(score)
        self.score += int(score)
        self.nums += 1
        return score

    def calculate_final(self):
        final_score = ((self.score - self.nums) / (self.nums * 4)) * 100
        self.score = 0
        self.nums = 0
        return final_score
    