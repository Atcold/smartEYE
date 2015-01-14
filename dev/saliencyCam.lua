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
--camRes  (default ok) Camera resolution (nHD|VGA|FWVGA|ok|HD|FHD)
--mode    (default 1)  Different mode
]])

-- Switch input sources
res = {
   nHD   = {w =  640, h =  360},
   VGA   = {w =  640, h =  480},
   FWVGA = {w =  854, h =  480},
   ok    = {w = 1024, h =  576},
   HD    = {w = 1280, h =  720},
   FHD   = {w = 1920, h = 1080},
}

cam = image.Camera.new{
   width  = res[opt.camRes].w,
   height = res[opt.camRes].h
}

-- Auxiliary functions ---------------------------------------------------------
image.fit = function (inputImage)
   return image.scale(inputImage:float(), img:size(3), img:size(2), 'simple')
end
mask = function (inputImage, mask, th)
   local temp
   if opt.mode == 1 then
      temp = mask:gt(th):float()[1]--:mul(.8):add(.2)
      temp = image.dilate(image.erode(temp, image.gaussian(10):gt(.2):float()), image.gaussian(20):gt(.2):float())
      temp = temp:mul(.8):add(.2):repeatTensor(3,1,1)
   elseif opt.mode == 2 then
      mask = mask[1]
      k = image.gaussian{size = 7, normalize = true}:float()
      mask = image.convolve(mask, k, 'same'):repeatTensor(3,1,1)
      temp = mask:gt(th):float():mul(.8):add(.2)
   elseif opt.mode == 3 then
      temp = mask:gt(th):float():mul(.8):add(.2)
   end

   return inputImage:clone():cmul(temp)
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
      zoom = 1, nrow = 2, scaleeach = true, win = win, min = 0, max = 1,
      legend = '"person" pseudo probability and top-down saliency map',
      image = {
         img:float(),
         image.fit(colorMap),
         mask(img:float(), gradManipulation(gradInput, 0), .1),
         --gradManipulation(gradInput, 0),
         gradManipulation(gradInput, 0),
      }
   }

   win.painter:gbegin()
   winSize = win.window.frameSize:totable()
   win.painter:setcolor(1,1,1)
   win.painter:setfontsize(30)
   win.painter:moveto(20,40+winSize.height/2)
   win.painter:show('saliency')
   win.painter:moveto(20+winSize.width/2,40+winSize.height/2)
   win.painter:show('saliency fine grain')
   win.painter:moveto(20+winSize.width/2,40)
   win.painter:show('heat-map')
   win.painter:setcolor(0,0,0)
   win.painter:moveto(20,40)
   win.painter:show('Camera input')
   win.painter:gend()

   return win.window.visible

end

-- Main program ----------------------------------------------------------------
running = true
while running do running = run() end
