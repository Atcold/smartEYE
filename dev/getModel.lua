--------------------------------------------------------------------------------
-- Get model cleaned and ready
--------------------------------------------------------------------------------
-- Alfredo Canziani, Nov 14
--------------------------------------------------------------------------------

require 'cunn'
local GPUcount = cutorch.getDeviceCount()
if GPUcount > 1 then
   io.write('Choose free GPU (1-' .. GPUcount .. ') ')
   cutorch.setDevice(io.read())
end

oldModel = torch.load('../net/17cate9filter/model-127.net')

-- Building new MLP
oldMLP = oldModel.modules[2]
MLP = nn.Sequential():cuda()
MLP:add(oldMLP.modules[1]) -- Reshape
-- 2 is Dropout
MLP:add(oldMLP.modules[3]) -- Linear
MLP:add(nn.ReLU():cuda()) -- 4
-- 5 is Dropout
MLP:add(oldMLP.modules[6]) -- Linear
MLP:add(nn.ReLU():cuda()) -- ReLU
-- 8 is Dropout
MLP:add(oldMLP.modules[9]) -- Linear

-- Creating new model
model = nn.Sequential():cuda()
model:add(oldModel.modules[1])
model:add(MLP)

-- Removing custom updateGradInput()
model.modules[1].modules[1].updateGradInput = nil

-- Creating SoftMax SM module
SM = nn.SoftMax():cuda()
LSM = nn.LogSoftMax():cuda()
LSMf = nn.LogSoftMax():float()
loss = nn.ClassNLLCriterion():cuda()

-- Cleaning
oldModel = nil
oldMLP = nil
collectgarbage()
