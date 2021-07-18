import torch

# CSRNet
#test = torch.load("C:/.lwcc/weights/CSRNet_SHB_old.pth.tar")
#state = dict()
#state["model"] = test["state_dict"]
#torch.save(state, "C:/.lwcc/weights/CSRNet_SHB.pth")

# SFANet
test = torch.load("C:/.lwcc/weights/SFANet_SHA_old.pth")
print(test.keys())
state = dict()
state["model"] = test["model"]
torch.save(state, "C:/.lwcc/weights/SFANet_SHA.pth")

#Bay
#test = torch.load("C:/.lwcc/weights/Bay_QNRF_old.pth")
#state = dict()
#state["model"] = test
#torch.save(state, "C:/.lwcc/weights/Bay_QNRF.pth")

# DM-Count
#test = torch.load("C:/.lwcc/weights/DM-Count_SHB_old.pth")
#state = dict()
#state["model"] = test
#torch.save(state, "C:/.lwcc/weights/DM-Count_SHB.pth")


#print(test)