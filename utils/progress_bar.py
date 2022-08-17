import sys 
import re 


class ProgressBar(object):
    DEFAULT = "Progress: %(bar)s %(percent)3d%%"
    def __init__(self, mode, epoch=None, total_epoch=None, current_loss=None, avg_loss=None, 
            current_loss1=None, avg_loss1=None, current_loss2=None, avg_loss2=None, cur_time=None, 
            model_name=None, total=None, current=None, 
            last_lr=0.0, width =35, symbol = ">", output=sys.stderr):
        assert len(symbol) == 1

        self.mode = mode
        self.total = total
        self.symbol = symbol
        self.output = output
        self.width = width
        self.current = current
        self.epoch = epoch
        self.total_epoch = total_epoch
        self.current_loss = current_loss
        self.avg_loss = avg_loss
        self.current_loss1 = current_loss1
        self.avg_loss1 = avg_loss1
        self.current_loss2 = current_loss2
        self.avg_loss2 = avg_loss2
        self.cur_time = cur_time 
        self.model_name = model_name
        self.last_lr = last_lr

    def __call__(self):
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        bar = "[" + self.symbol * size + " " * (self.width - size) + "]"

        args = {
            "mode": self.mode,
            "total": self.total,
            "bar" : bar,
            "current": self.current,
            "percent": percent * 100,
            "current_loss": self.current_loss,
            "avg_loss" : self.avg_loss,
            "xtr_current_loss": self.current_loss1,
            "xtr_avg_loss" : self.avg_loss1,
            "sentalign_current_loss": self.current_loss2,
            "sentalign_avg_loss" : self.avg_loss2,
            "cur_time": self.cur_time,
            "epoch": self.epoch,
            "epochs": self.total_epoch,
            "last_lr": self.last_lr
        }
        message = "\033[1;32;40m %(cur_time)s-%(mode)s Epoch:%(epoch)d/%(epochs)d %(bar)s\033[0m [ Current Loss: %(current_loss).3f Avg Loss: %(avg_loss).3f || XTR Current Loss: %(xtr_current_loss).3f XTR Avg Loss: %(xtr_avg_loss).3f || Align Current Loss: %(sentalign_current_loss).3f Align Avg Loss: %(sentalign_avg_loss).3f || Current LR: %(last_lr).6f ]  %(current)d/%(total)d \033[1;32;40m[ %(percent)3d%% ]\033[0m" % args
        # self.write_message = " %(cur_time)s-%(mode)s Epoch:%(epoch)d/%(epochs)d %(bar)s [ Current Loss: %(current_loss).3f Avg Loss: %(avg_loss).3f XTR Current Loss: %(xtr_current_loss).3f XTR Avg Loss: %(xtr_avg_loss).3f Align Current Loss: %(sentalign_current_loss).3f Align Avg Loss: %(sentalign_avg_loss).3f Current LR: %(last_lr).6f ]  %(current)d/%(total)d [ %(percent)3d%% ]" % args
        print(message + '\n', file=self.output, end="")
        

    def done(self):
        self.current = self.total
        self()
        # print("", file=self.output)

if __name__ == "__main__":
    from time import sleep
    progress = ProgressBar("Train",total_epoch=2,model_name="resnet159")
    for i in range(2):
        progress.total = 50
        progress.epoch = i
        progress.current_loss = 0.15
        progress.avg_loss = 0.15
        for x in range(2):
            progress.current = x
            progress()
            sleep(0.1)
        progress.done()

