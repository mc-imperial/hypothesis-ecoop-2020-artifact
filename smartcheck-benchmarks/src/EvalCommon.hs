{-# LANGUAGE CPP #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE DeriveDataTypeable #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}

module EvalCommon where

import Test.QuickCheck
import Test.QuickCheck.Random
import System.Environment


#ifdef USE_SMARTCHECK

import Test.SmartCheck

scArgs seed = scStdArgs {
    qcArgs  = quickcheckArgs seed
    , format  = PrintString
    , runForall   = False
    , runExists   = False
}
#endif

large = 1000000

quickcheckArgs seed = stdArgs{
    replay=Just (mkQCGen seed, 0), maxSuccess=large
#if MIN_VERSION_QuickCheck(2, 11, 3)
    , maxShrinks=large
#endif
} 

evalQuickCheck prop = do seedString <- getEnv "SEED"
                         envShrink <- lookupEnv "SHRINK"
                         let shrink = envShrink /= (Just "false")
                         let seed = read seedString
#ifdef USE_SMARTCHECK
                         smartCheck (scArgs seed) prop
#else
                         verboseCheckWith (quickcheckArgs seed) prop
#endif
