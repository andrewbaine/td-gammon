import asyncio


async def readlines(reader, chunk_size=1024):
    chunks = []
    sep = b"\0"
    while True:
        data = await reader.read(chunk_size)
        print("data", data)
        if not data:
            return
        i = 0
        while i < len(data):
            end = data.find(sep)
            if end < 0:
                chunks.append(data)
                break
            else:
                c = data[i:end]
                chunks.append(c)
                yield b"".join(chunks)
                chunks = []
                i = end + 1


async def main(host, port, f):
    async def handler(reader, writer):
        async for data in readlines(reader):
            response = f(data.decode())
            if response:
                writer.write(response.encode())
                await writer.drain()
        writer.close()

    server = await asyncio.start_server(handler, host, port)
    async with server:
        await server.serve_forever()
