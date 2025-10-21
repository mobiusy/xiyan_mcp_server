
from .server import mcp, global_config, mcp_config

def main():
    mcp.run(transport=mcp_config.get('transport', 'streamable-http'))



if __name__ == "__main__":
    main()