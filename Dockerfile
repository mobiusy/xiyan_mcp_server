# 使用Python 3.11作为基础镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 复制requirements.txt文件
COPY requirements.txt ./

# 安装依赖
RUN pip install -r requirements.txt
RUN pip install psycopg2-binary

# 复制项目代码
COPY . ./

WORKDIR /app/src

# 运行应用
CMD ["python", "-m", "xiyan_mcp_server"]