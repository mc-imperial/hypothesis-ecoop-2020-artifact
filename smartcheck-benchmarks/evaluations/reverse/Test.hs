module Main where

import EvalCommon

prop_reverse :: [Int] -> Bool
prop_reverse ls = reverse ls == ls

main = evalQuickCheck prop_reverse
