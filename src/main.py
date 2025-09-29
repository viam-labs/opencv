import asyncio
from viam.module.module import Module
try:
    from models.chessboard import Chessboard
except ModuleNotFoundError:
    # when running as local module with run.sh
    from .models.chessboard import Chessboard


if __name__ == '__main__':
    asyncio.run(Module.run_from_registry())
