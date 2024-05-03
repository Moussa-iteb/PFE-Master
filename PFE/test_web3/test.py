
from web3 import Web3
#from web3.auto.gethdev import w3

import json



truffle_path = 'C:/Users/ACER/PFE/test_web3/'
truffle_file = json.load(open(truffle_path + 'earn_project.json'))
abi=truffle_file['abi']
bytecode= truffle_file['bytecode']
we3 = Web3 (Web3.HTTPProvider(" http://localhost:9545"))
contract = we3.eth.contract(abi=abi, address='0xabEfC51AA58b31E46b00872911020Ca384B92DC6')
#we3.middleware_onion.inject(geth_poa_middleware , layer=0)

#we3.eth.defaultAccount = we3.eth.account.from_key('65b8cee02d4b85ed39dac39f9b6d0902af56ff194d8ff99e2ae2de1a7e308654').address
we3.eth.defaultAccount = we3.eth.account.from_key('0be4cc97daabdd7c451bb27acb494cacd029525ac4aeb1e6f719b792203df1a0').address

#we3.eth.defaultAccount = '0x25b6ED33F222C89944c49B3489f5D388555386eD'
print(we3.eth.get_balance(we3.eth.defaultAccount))
print(we3.eth.defaultAccount)
print(we3.is_connected())
#w3.eth.defaultAccount= w3.eth.accounts[2]
valuetx=contract.functions.getDataByDate(123).call()
print(valuetx)
