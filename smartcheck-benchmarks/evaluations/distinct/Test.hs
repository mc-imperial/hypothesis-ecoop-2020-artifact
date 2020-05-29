module Main where

import EvalCommon

import Data.List (nub)

prop_distinct :: [Int] -> Bool
prop_distinct ls = length (nub ls) < 3

main = evalQuickCheck prop_distinct
