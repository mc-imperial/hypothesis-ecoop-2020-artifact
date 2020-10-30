{-# LANGUAGE CPP #-}

{-# LANGUAGE ScopedTypeVariables, TemplateHaskell, DeriveDataTypeable, StandaloneDeriving #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveDataTypeable #-}

{-# OPTIONS_GHC -fno-warn-missing-signatures #-}
{-# OPTIONS_GHC -fno-warn-orphans #-}

{-
This test is derived from a mix of QuickCheck2's test and SmartCheck's
regression suite.
https://raw.githubusercontent.com/nick8325/quickcheck/master/examples/Heap.hs
https://github.com/leepike/SmartCheck/blob/master/regression/Heap/Heap_Program.hs

Primarily the former, as the latter did a bunch of stuff with generating
heap programs and then immediately threw it away.

Notable differences:
  * we have added a precondition to the test that the heap invariant must be
    satisfied. This prevents SmartCheck winning by massively by generating
    invalid test cases.
  * We have enabled QuickChecks' genericShrink for Heap values, as previously
    it had no shrinking at all.  
-}

module Main where

--------------------------------------------------------------------------
-- imports

import System.Environment

import EvalCommon
import Test.QuickCheck
import Test.QuickCheck.Poly

import Control.Monad
import Data.List (sort)
import Data.Typeable

import GHC.Generics

#ifdef USE_SMARTCHECK
import Test.SmartCheck
import qualified Test.SmartCheck as SC
#endif

--------------------------------------------------------------------------
deriving instance Typeable OrdA
deriving instance Generic OrdA

instance Read OrdA where
  readsPrec i = \s ->
    let rd = readsPrec i s :: [(Integer,String)] in
    let go (i',s') = (OrdA i', s') in
    map go rd


#ifdef USE_SMARTCHECK
instance SC.SubTypes OrdA
instance (SC.SubTypes a, Ord a, Arbitrary a, Generic a)
         => SC.SubTypes (Heap a)
#endif

instance (Ord a, Arbitrary a, Generic a) => Arbitrary (Heap a) where
  arbitrary = sized (arbHeap Nothing)
   where
    arbHeap mx n =
      frequency $
        [ (1, return Empty) ] ++
        [ (7, do my <- arbitrary `suchThatMaybe` ((>= mx) . Just)
                 case my of
                   Nothing -> return Empty
                   Just y  -> liftM2 (Node y) arbHeap2 arbHeap2
                    where arbHeap2 = arbHeap (Just y) (n `div` 2))
        | n > 0
        ]
#ifndef USE_SMARTCHECK
  shrink = genericShrink
#endif


data Heap a
  = Node a (Heap a) (Heap a)
  | Empty
 deriving ( Eq, Ord, Show, Generic)

toList :: Heap a -> [a]
toList h = toList' [h]
 where
  toList' []                  = []
  toList' (Empty        : hs) = toList' hs
  toList' (Node x h1 h2 : hs) = x : toList' (h1:h2:hs)

toSortedList :: Ord a => Heap a -> [a]
toSortedList Empty          = []
toSortedList (Node x h1 h2) = x : toList (h1 `merge` h2)

invariant :: Ord a => Heap a -> Bool
invariant Empty          = True
invariant (Node x h1 h2) = x <=? h1 && x <=? h2 && invariant h1 && invariant h2

(<=?) :: Ord a => a -> Heap a -> Bool
x <=? Empty      = True
x <=? Node y _ _ = x <= y

(==?) :: Ord a => Heap a -> [a] -> Bool
h ==? xs = invariant h && sort (toList h) == sort xs

prop_ToSortedList (h :: Heap OrdA) =
  invariant h ==> h ==? xs && xs == sort xs
 where
  xs = toSortedList h


merge :: Ord a => Heap a -> Heap a -> Heap a
h1    `merge` Empty = h1
Empty `merge` h2    = h2
h1@(Node x h11 h12) `merge` h2@(Node y h21 h22)
  | x <= y    = Node x (h12 `merge` h2) h11
  | otherwise = Node y (h22 `merge` h1) h21

main = evalQuickCheck prop_ToSortedList
