import sys
from train.process_ckpt import extract_small_model

path = str(sys.argv[1])
name = str(sys.argv[2])
sample_rate = str(sys.argv[3])
if_f0 = int(sys.argv[4])
if(len(sys.argv) == 5):
    info = ""
else:
    info = str(sys.argv[5])

print("Mangio-RVC-Fork Small Model Extraction: Performing extraction...")
extraction = extract_small_model(path, name, sample_rate, if_f0, info)
print("Mangio-RVC-Fork Small Model Extraction: %s" % (extraction))
if(extraction == "Success."):
    print("Mangio-RVC-Fork Small Model Extraction: Placed %s.pth into ./weights" % (name))


