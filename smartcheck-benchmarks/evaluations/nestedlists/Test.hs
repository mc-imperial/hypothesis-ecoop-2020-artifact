module Main where

import EvalCommon

prop_nestedSmall :: [[()]] -> Bool
prop_nestedSmall ls = (sum . map length $ ls) <= 10

main = evalQuickCheck prop_nestedSmall
