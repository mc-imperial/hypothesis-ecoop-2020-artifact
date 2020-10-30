module Main where

import EvalCommon
import Test.QuickCheck
import Data.List

prop_delete :: ([Int], Int) -> Property
prop_delete (ls, i) = 0 <= i && i < length ls ==> not (elem x (delete x ls))
    where x = ls !! i

main = evalQuickCheck prop_delete
