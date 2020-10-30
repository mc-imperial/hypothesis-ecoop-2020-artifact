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

newtype SmallInt = SmallInt Integer deriving (Eq, Generic, Typeable)


instance Show SmallInt where
    show (SmallInt n) = show n


instance Arbitrary SmallInt where
  arbitrary = SmallInt <$> choose (0, 10 ^ 6)
#ifndef USE_SMARTCHECK
  shrink (SmallInt s) = [SmallInt s' | s' <- shrink s, (0 <= s') && (s' <= 10 ^ 6)]
#endif

newtype VSMall = VSMall Integer deriving (Eq, Generic, Typeable)


instance Show VSMall where
    show (VSMall n) = show n


instance Arbitrary VSMall where
  arbitrary = VSMall <$> choose (2, 5)
#ifndef USE_SMARTCHECK
  shrink (VSMall s) = [VSMall s' | s' <- shrink s, (2 <= s') && (s' <= 5)]
#endif

#ifdef USE_SMARTCHECK
instance SC.SubTypes VSMall
instance SC.SubTypes SmallInt
#endif

prods :: (SmallInt, [VSMall]) -> [Integer]
prods ((SmallInt n), []) = [n]
prods ((SmallInt n), ((VSMall x):xs)) = n : prods (SmallInt $ n * x,  xs)


prop_reverse :: (SmallInt, [VSMall]) -> Bool
prop_reverse = not . any (>= 10 ^ 6) . prods

main = evalQuickCheck prop_reverse
