import os
import sys
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from cdp_langchain.agent_toolkits import CdpToolkit
from cdp_langchain.utils import CdpAgentkitWrapper

load_dotenv()
app = FastAPI()


class FundingRecord(BaseModel):
    user_id: str
    wallet_address: str

@app.post("/api/register-wallet")
async def register_wallet(funding: FundingRecord):
    """Register user's wallet address when they first fund CDP wallet"""
    try:
        # Save user's wallet address
        with open(f"user_wallets/{funding.user_id}_funding.json", "w") as f:
            json.dump({
                "wallet_address": funding.wallet_address,
                "timestamp": datetime.now().isoformat()
            }, f)
        
        return {"status": "success", "message": "Wallet registered"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class WithdrawRequest(BaseModel):
    user_id: str
    token: str
    amount: float

@app.post("/api/withdraw")
async def withdraw_tokens(withdraw_request: WithdrawRequest):
    """Withdraw tokens back to user's original wallet"""
    try:
        # Get user's registered wallet address
        try:
            with open(f"user_wallets/{withdraw_request.user_id}_funding.json", "r") as f:
                funding_info = json.load(f)
                user_wallet_address = funding_info["wallet_address"]
        except FileNotFoundError:
            raise HTTPException(
                status_code=400,
                detail="No registered wallet found. Please fund CDP wallet first"
            )

        # Initialize CDP wallet
        user_wallet = UserWallet(withdraw_request.user_id)
        agent_executor, config = await user_wallet.initialize_agent()
        
        # Verify token balance
        balance_query = f"What is my {withdraw_request.token} balance?"
        token_balance = 0.0
        
        for chunk in agent_executor.stream(
            {"messages": [HumanMessage(content=balance_query)]},
            config
        ):
            if "tools" in chunk:
                response = chunk["tools"]["messages"][0].content
                try:
                    import re
                    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", response)
                    if numbers:
                        token_balance = float(numbers[0])
                except:
                    pass

        if token_balance < withdraw_request.amount:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient balance. Have: {token_balance} {withdraw_request.token}"
            )

        # Execute withdrawal to user's registered wallet
        transfer_instruction = (
            f"Transfer {withdraw_request.amount} {withdraw_request.token} "
            f"to address: {user_wallet_address}. "
            f"Wait for confirmation and show transaction hash."
        )

        result = {
            "status": "pending",
            "tx_hash": None,
            "amount": withdraw_request.amount,
            "token": withdraw_request.token,
            "to_address": user_wallet_address
        }

        for chunk in agent_executor.stream(
            {"messages": [HumanMessage(content=transfer_instruction)]},
            config
        ):
            if "tools" in chunk:
                response = chunk["tools"]["messages"][0].content
                if "0x" in response:
                    result["tx_hash"] = "0x" + response.split("0x")[1].split()[0]
                if "success" in response.lower():
                    result["status"] = "success"

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class SwapRequest(BaseModel):
    user_id: str
    tokens: List[str]
    amounts: List[float]

class UserWallet:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.wallet_dir = "user_wallets"
        self.wallet_file = f"{self.wallet_dir}/{user_id}_wallet.txt"
        os.makedirs(self.wallet_dir, exist_ok=True)

    async def verify_balance(self, agent_executor, config) -> Dict:
        """Verify current wallet balance"""
        balance_info = {
            "balance_eth": 0.0,
            "verified": False,
            "last_checked": datetime.now().isoformat()
        }

        balance_query = "Show my exact ETH balance as a number"
        for chunk in agent_executor.stream(
            {"messages": [HumanMessage(content=balance_query)]},
            config
        ):
            if "tools" in chunk:
                response = chunk["tools"]["messages"][0].content
                try:
                    import re
                    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", response)
                    if numbers:
                        balance = float(numbers[0])
                        if balance >= 0:
                            balance_info["balance_eth"] = balance
                            balance_info["verified"] = True
                except Exception as e:
                    print(f"Balance parsing error: {e}")
                    balance_info["error"] = str(e)

        return balance_info

    async def initialize_agent(self):
        """Initialize CDP agent"""
        llm = ChatOpenAI(model="gpt-4-turbo-preview")
        
        wallet_data = None
        if os.path.exists(self.wallet_file):
            with open(self.wallet_file) as f:
                wallet_data = f.read()
        
        values = {
            "network_id": "base-mainnet",
            "cdp_wallet_data": wallet_data if wallet_data else None
        }
        
        agentkit = CdpAgentkitWrapper(**values)
        
        wallet_data = agentkit.export_wallet()
        with open(self.wallet_file, "w") as f:
            f.write(wallet_data)
        
        cdp_toolkit = CdpToolkit.from_cdp_agentkit_wrapper(agentkit)
        tools = cdp_toolkit.get_tools()
        
        memory = MemorySaver()
        config = {"configurable": {"thread_id": f"User-{self.user_id}-Agent"}}
        
        agent = create_react_agent(
            llm,
            tools=tools,
            checkpointer=memory,
            state_modifier=(
                "You are a CDP trading assistant on Base mainnet. "
                "Use wallet.trade(amount, 'eth', token) for trades. "
                "Always verify balance before trading. "
                "Show all transaction details and wait for confirmations."
            )
        )
        
        return agent, config

    async def execute_swap(self, agent_executor, config, token: str, amount: float) -> Dict:
        """Execute single token swap"""
        result = {
            "token": token,
            "amount": amount,
            "status": "pending",
            "tx_hash": None,
            "error": None
        }

        # Execute swap with explicit instructions
        swap_instruction = (
            f"Follow these steps exactly:\n"
            f"1. Trade {amount} ETH for {token} using: trade = wallet.trade({amount}, 'eth', '{token.lower()}')\n"
            f"2. Wait for confirmation using: trade.wait()\n"
            f"3. Show the transaction hash\n"
            f"4. Confirm the trade completed"
        )

        print(f"\nExecuting swap: {amount} ETH → {token}")
        
        for chunk in agent_executor.stream(
            {"messages": [HumanMessage(content=swap_instruction)]},
            config
        ):
            if "tools" in chunk:
                response = chunk["tools"]["messages"][0].content
                print(f"Response: {response}")
                
                if "0x" in response:
                    try:
                        result["tx_hash"] = "0x" + response.split("0x")[1].split()[0]
                    except:
                        pass
                
                if "success" in response.lower() or "confirmed" in response.lower():
                    result["status"] = "success"
                elif "error" in response.lower() or "reverted" in response.lower():
                    result["status"] = "failed"
                    result["error"] = response

        return result

@app.post("/api/batch-swap")
async def batch_swap(swap_request: SwapRequest):
    """Execute batch swap with tracking"""
    try:
        if len(swap_request.tokens) != len(swap_request.amounts):
            raise HTTPException(
                status_code=400,
                detail="Tokens and amounts must have same length"
            )
        
        if len(swap_request.tokens) > 2:
            raise HTTPException(
                status_code=400,
                detail="Maximum 2 tokens allowed per batch"
            )
        
        # Initialize user's wallet
        user_wallet = UserWallet(swap_request.user_id)
        agent_executor, config = await user_wallet.initialize_agent()
        
        # Check current balance
        balance_info = await user_wallet.verify_balance(agent_executor, config)
        if balance_info["balance_eth"] < sum(swap_request.amounts):
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient balance. Have: {balance_info['balance_eth']} ETH, Need: {sum(swap_request.amounts)} ETH"
            )
        
        # Execute swaps
        results = []
        for token, amount in zip(swap_request.tokens, swap_request.amounts):
            print(f"\nProcessing swap: {amount} ETH → {token}")
            result = await user_wallet.execute_swap(agent_executor, config, token, amount)
            results.append(result)
        
        # Get final balance
        final_balance = await user_wallet.verify_balance(agent_executor, config)
        
        return {
            "user_id": swap_request.user_id,
            "timestamp": datetime.now().isoformat(),
            "balance_before": balance_info["balance_eth"],
            "balance_after": final_balance["balance_eth"],
            "swaps": results,
            "status": "completed"
        }
        
    except Exception as e:
        print(f"Error in batch_swap: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@app.get("/api/wallet/{user_id}")
async def get_wallet_info(user_id: str):
    """Get wallet information"""
    try:
        user_wallet = UserWallet(user_id)
        agent_executor, config = await user_wallet.initialize_agent()
        
        # Get balance
        balance_info = await user_wallet.verify_balance(agent_executor, config)
        
        # Get address
        address_query = "What is my CDP wallet address?"
        address = None
        for chunk in agent_executor.stream(
            {"messages": [HumanMessage(content=address_query)]},
            config
        ):
            if "tools" in chunk:
                response = chunk["tools"]["messages"][0].content
                if "0x" in response:
                    addr = response.split("0x")[1].split()[0].strip(":,")
                    address = f"0x{addr}"
        
        return {
            "user_id": user_id,
            "address": address,
            "balance": balance_info,
            "wallet_file": f"user_wallets/{user_id}_wallet.txt",
            "is_new_wallet": not os.path.exists(f"user_wallets/{user_id}_wallet.txt"),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)