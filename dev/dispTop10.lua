--------------------------------------------------------------------------------
-- Display (or save) the top10 test images
--------------------------------------------------------------------------------
-- Alfredo Canziani, Nov 14
--------------------------------------------------------------------------------

require 'image'
require 'ext'

for label, cls in pairs(top10) do
   -- Display
   image.display{image = cls.image, legend = label, nrow = 5}

   -- Save
   -- image.saveJPG(label .. '.jpg',image.toDisplayTensor{
   --    input = cls.image, legend = label, nrow = 5
   -- })
end
