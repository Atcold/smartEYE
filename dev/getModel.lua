--------------------------------------------------------------------------------
-- Get model cleaned and ready
--------------------------------------------------------------------------------
-- Alfredo Canziani, Nov 14
--------------------------------------------------------------------------------

require 'cunn'

oldModel = torch.load('17cate9filter/model-127.net')

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

-- Removing old one
oldModel = nil
oldMLP = nil

-- Creating SoftMax SM module
SM = nn.SoftMax():cuda()
LSM = nn.LogSoftMax():cuda()
LSMf = nn.LogSoftMax():float()
loss = nn.ClassNLLCriterion():cuda()

-- Loading images and classes, building reverse classes
top10 = torch.load('Top10TestData.t7')
classes = torch.load('classes.t7')
revClas = {}; for a,b in ipairs(classes) do revClas[b] = a end
