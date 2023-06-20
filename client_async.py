import asyncio

import aiohttp_sse_client.client
from aiohttp import ClientSession
from aiohttp_sse_client import client as sseclient


async def handle_event(event: aiohttp_sse_client.client.MessageEvent, event_source):
    # 处理 SSE 事件的回调函数
    print(f'Event type: {event.type}')
    print(f'Event id: {event.last_event_id}')
    print(f'Event data: {event.data}')
    print(f'Event message: {event.message}')
    if event.type == "finish":
        try:
            await event_source.close()
        except Exception as err:
            print("close with error", err)


async def listen_sse():
    async with ClientSession() as session:
        url = 'http://localhost:8000/stream_chat'
        data = {"prompt": "你好", "history": []}
        headers = {'Content-Type': 'application/json'}
        async with sseclient.EventSource(url, json=data, headers=headers, session=session) as event_source:
            try:
                async for event in event_source:
                    # 将事件传递给回调函数进行处理
                    await handle_event(event, event_source)
            except Exception as err:
                print("event close", err)
asyncio.run(listen_sse())