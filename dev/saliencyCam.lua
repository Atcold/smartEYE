--------------------------------------------------------------------------------
-- Generate saliency map from camera image (demo)
--------------------------------------------------------------------------------
-- Alfredo Canziani, Jan 15
--------------------------------------------------------------------------------

require 'camera'
require 'imgraph'
require 'getSpatialModel'
require 'pl'

-- Options ---------------------------------------------------------------------
opt = lapp([[
--camRes  (default HD) Camera resolution (QHD|VGA|FWVGA|HD|FHD)
]])

-- Switch input sources
res = {
   QHD   = {w =  640, h =  360},
   VGA   = {w =  640, h =  480},
   FWVGA = {w =  854, h =  480},
   HD    = {w = 1280, h =  720},
   FHD   = {w = 1920, h = 1080},
}

cam = image.Camera.new{
   width  = res[opt.camRes].w,
   height = res[opt.camRes].h
}

image.fit = function (inputImage)
   return image.scale(inputImage:float(), img:size(3), img:size(2), 'simple')
end
mask = function (map, mask, th)
   return map:clone():cmul(mask:float():repeatTensor(3,1,1):gt(th):float())
end
gradManipulation = function (grad, th)
   return grad:float():abs():max(1):repeatTensor(3,1,1)
end

run = function ()

   img = cam:forward():cuda()
   model:forward(img:cuda())

   -- Estimating pseudo-probability
   psProb = torch.CudaTensor(#model.output)

   -- This has to be rewritten with FFI
   for i = 1, model.output:size(2) do
      for j = 1, model.output:size(3) do
         psProb[{ {},i,j }] = SM:forward(model.output[{ {},i,j }])
      end
   end

   colorMap = imgraph.colorize(psProb[2]:float()*255,image.jetColormap(256):float())

   def = psProb:clone():zero()
   def[2] = psProb[2]:clone():mul(-1)
   gradInput = model:updateGradInput(img:cuda(), def)
   -- image.display{image = gradInput:float(), zoom = 0.5}
   win = image.display{
      zoom = 1, nrow = 2, scaleeach = true, win = win,
      legend = '"person" pseudo probability and top-down saliency map',
      image = {
         img:float(),
         image.fit(colorMap),
         gradInput:float(),
         gradManipulation(gradInput, 0)
      }
   }

   return win.window.visible

end

running = true
while running do running = run() end
