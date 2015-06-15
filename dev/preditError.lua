--------------------------------------------------------------------------------
-- Predict error for a specific image
--------------------------------------------------------------------------------
-- Alfredo Canziani, Nov 14
--------------------------------------------------------------------------------

require 'cunn'
require 'ext'

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
--MLP:add(nn:SoftMax():cuda())
MLP:add(nn:LogSoftMax():cuda())

model = nn.Sequential():cuda()
model:add(oldModel.modules[1])
model:add(MLP)

-- Disabling <dropout> on oldModel
oldMLP.modules[2].train = false
oldMLP.modules[5].train = false
oldMLP.modules[8].train = false
LSM = nn.LogSoftMax():float() -- explains the discrepancy in the predictions values

crit = nn.ClassNLLCriterion():cuda()

for label, class in pairs(top10) do
   print('Prediction for ' .. label)
   for n, img in ipairs(class.image) do
      newE = crit:forward(model:forward(img:cuda()), revClas[label])
      oldE = crit:forward(LSM:forward(oldModel:forward(img:cuda()):float()), revClas[label])
      print(string.format(
         'New pred E: %.5f, old pred E: %.5f, saved: %.5f',
         newE, oldE, class.error[n]
      ))
   end
   io.write("Press enter for next category, 'q' for quitting... ")
   if io.read() == 'q' then break end
end

