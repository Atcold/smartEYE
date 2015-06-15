--------------------------------------------------------------------------------
-- Spatialise a MLP
--------------------------------------------------------------------------------
-- Alfredo Canziani, Nov 14, Feb 15
--------------------------------------------------------------------------------

require 'sys'

if sys.execute('uname -a'):find('tegra') then
   require 'cudnn'
   model = torch.load('../net/17cate9filter/model-127.net.tegra'):cuda()
   SM = cudnn.SoftMax():cuda()
else

   -- Getting model and renaming
   require 'getModel'

   oldModel = model
   oldMLP = oldModel.modules[2]

   -- Defining new MLP
   newMLP = nn.Sequential()
   newMLP:add(nn.SpatialConvolutionMM(  32, 2048, 6, 6))
   newMLP:add(nn.ReLU())
   newMLP:add(nn.SpatialConvolutionMM(2048, 2048, 1, 1))
   newMLP:add(nn.ReLU())
   newMLP:add(nn.SpatialConvolutionMM(2048,   17, 1, 1))

   -- Sending MLP to Cuda
   newMLP:cuda()

   -- Copying over the weights from oldMLP
   -- Kernels
   newMLP.modules[1].weight:copy(oldMLP.modules[2].weight)
   newMLP.modules[3].weight:copy(oldMLP.modules[4].weight)
   newMLP.modules[5].weight:copy(oldMLP.modules[6].weight)
   -- Bias
   newMLP.modules[1].bias:copy(oldMLP.modules[2].bias)
   newMLP.modules[3].bias:copy(oldMLP.modules[4].bias)
   newMLP.modules[5].bias:copy(oldMLP.modules[6].bias)

   -- Creating new final model WITH preprocessing
   model = nn.Sequential():cuda()
   model:add(nn.AddConstant(-0.4124)):cuda()
   model:add(nn.MulConstant(1/0.2805)):cuda()
   model:add(oldModel.modules[1])
   model:add(newMLP)

   -- Cleaning
   oldModel = nil
   oldMLP = nil
   collectgarbage()

end

-- Creating a SpatialSoftMax function
nn.SpatialSoftMax = function (spatialLogit)
   return SM(
      spatialLogit
         :reshape(spatialLogit:size(1), spatialLogit:size(2) * spatialLogit:size(3))
         :t():contiguous()
      ):t():reshape(spatialLogit:size())
end
