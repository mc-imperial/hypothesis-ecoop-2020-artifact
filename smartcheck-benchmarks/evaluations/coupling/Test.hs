{-# LANGUAGE CPP #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveDataTypeable #-}

module Main where

import EvalCommon
import Test.QuickCheck
import GHC.Generics
import Data.Typeable

#ifdef USE_SMARTCHECK
import Test.SmartCheck
import qualified Test.SmartCheck as SC
#endif

newtype SmallInt = SmallInt Int deriving (Eq, Generic, Typeable)


instance Show SmallInt where
    show (SmallInt n) = show n


instance Arbitrary SmallInt where
  arbitrary = SmallInt <$> choose (0, 10)
#ifndef USE_SMARTCHECK
  shrink = genericShrink
#endif

#ifdef USE_SMARTCHECK
instance SC.SubTypes SmallInt
#endif

enumerate :: [a] -> [(Int, a)]
enumerate ls = go 0 ls
    where go _ [] = []
          go n (x:xs) = (n, x) : go (n + 1) xs


prop_reverse :: [SmallInt] -> Property
prop_reverse ls = valid ==> not loopy
    where n = length ls
          loopy = or [(j /= i) && (ls !! j == SmallInt i) | (i, SmallInt j) <- enumerate ls]
          valid = and [(0 <= v) && (v < n) | SmallInt v <- ls]

main = evalQuickCheck prop_reverse
