--------------------------------------------------------------------------------
-- Get top 10 predictions per class
--------------------------------------------------------------------------------
-- Alfredo Canziani, Nov 14
--------------------------------------------------------------------------------

data = torch.load('../data/17cate9filter/TestDataset.t7')
cls = data.classes
data.classes = nil
top10data = {}

for _, c in pairs(cls) do
   clsE = torch.Tensor(data[c].error)
   e, idx = clsE:sort()
   for i = 1, 10 do
      if i == 1 then
         top10data[c] = {}
         top10data[c].image = {}
         top10data[c].error = {}
      end
      table.insert(top10data[c].image, data[c].image[idx[i]])
      table.insert(top10data[c].error, e[i])
   end
end

torch.save('../data/17cate9filter/Top10TestData.t7', top10data)
