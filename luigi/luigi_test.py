from luigi import run, Task, LocalTarget
import speed_segmentation as ss

class SpeedPattern(Task):
    
    filename = '../data/result/speed_patterns.csv'
    
    def run(self):
        ss.main(self.filename)
        
        
    def output(self):
        return LocalTarget(self.filename)
    

if __name__ == '__main__':
    run()