--------------------------------------------------------------------------------
-- Get ext-ernal variables
--------------------------------------------------------------------------------
-- Alfredo Canziani, Jun 15
--------------------------------------------------------------------------------

-- Loading images and classes, building reverse classes
top10 = torch.load('../data/17cate9filter/Top10TestData.t7')
classes = torch.load('../net/17cate9filter/classes.t7')
revClas = {}; for a,b in ipairs(classes) do revClas[b] = a end
