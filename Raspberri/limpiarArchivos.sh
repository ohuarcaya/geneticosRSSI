#!/bin/bash
find ./Tx_0x07 -name \p* -exec sed -i "s/\ \ \ \ bdaddr/\ /g" {} \;
find ./Tx_0x07 -name \p* -exec sed -i "s/\ \ \ \ RSSI:/\ /g" {} \;
find ./Tx_0x07 -name \p* -exec sed -i "s/(Random)/\ /g" {} \;
find ./Tx_0x07 -name \p* -exec sed -i "s/(Public)/\ /g" {} \;
