from release.aigcmn import AiGcMn
import torch
from torchvision.utils import save_image

aigcmn = AiGcMn('./release/nets/generator.pth')
labels = [1, 1, 4, 5, 1, 4]
labels = torch.Tensor(labels)
gen_output = aigcmn.generate(labels)

# 保存tensor
gen_output_numpy = gen_output.detach().numpy()
gen_output_numpy.tofile('./output/tensor.csv', sep=',')
print("Saved csv file!")

# 保存图片
save_image(gen_output, 'output/output.png')
print("Successfully saved output.")
