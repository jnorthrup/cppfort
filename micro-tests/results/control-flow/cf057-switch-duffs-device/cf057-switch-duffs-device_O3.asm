
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf057-switch-duffs-device/cf057-switch-duffs-device_O3.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z17test_duffs_devicePiPKii>:
100000360: 31001c48    	adds	w8, w2, #0x7
100000364: 11003849    	add	w9, w2, #0xe
100000368: 1a88b128    	csel	w8, w9, w8, lt
10000036c: 13037d08    	asr	w8, w8, #3
100000370: 6b0203e9    	negs	w9, w2
100000374: 12000929    	and	w9, w9, #0x7
100000378: 1200084a    	and	w10, w2, #0x7
10000037c: 5a894549    	csneg	w9, w10, w9, mi
100000380: 71000d3f    	cmp	w9, #0x3
100000384: 5400016c    	b.gt	0x1000003b0 <__Z17test_duffs_devicePiPKii+0x50>
100000388: 7100053f    	cmp	w9, #0x1
10000038c: 5400020c    	b.gt	0x1000003cc <__Z17test_duffs_devicePiPKii+0x6c>
100000390: 34000329    	cbz	w9, 0x1000003f4 <__Z17test_duffs_devicePiPKii+0x94>
100000394: 7100053f    	cmp	w9, #0x1
100000398: 54000521    	b.ne	0x10000043c <__Z17test_duffs_devicePiPKii+0xdc>
10000039c: b8404429    	ldr	w9, [x1], #0x4
1000003a0: b8004409    	str	w9, [x0], #0x4
1000003a4: 71000508    	subs	w8, w8, #0x1
1000003a8: 5400026c    	b.gt	0x1000003f4 <__Z17test_duffs_devicePiPKii+0x94>
1000003ac: 14000024    	b	0x10000043c <__Z17test_duffs_devicePiPKii+0xdc>
1000003b0: 7100153f    	cmp	w9, #0x5
1000003b4: 5400016c    	b.gt	0x1000003e0 <__Z17test_duffs_devicePiPKii+0x80>
1000003b8: 7100113f    	cmp	w9, #0x4
1000003bc: 540002c0    	b.eq	0x100000414 <__Z17test_duffs_devicePiPKii+0xb4>
1000003c0: 7100153f    	cmp	w9, #0x5
1000003c4: 54000240    	b.eq	0x10000040c <__Z17test_duffs_devicePiPKii+0xac>
1000003c8: 1400001d    	b	0x10000043c <__Z17test_duffs_devicePiPKii+0xdc>
1000003cc: 7100093f    	cmp	w9, #0x2
1000003d0: 540002a0    	b.eq	0x100000424 <__Z17test_duffs_devicePiPKii+0xc4>
1000003d4: 71000d3f    	cmp	w9, #0x3
1000003d8: 54000220    	b.eq	0x10000041c <__Z17test_duffs_devicePiPKii+0xbc>
1000003dc: 14000018    	b	0x10000043c <__Z17test_duffs_devicePiPKii+0xdc>
1000003e0: 7100193f    	cmp	w9, #0x6
1000003e4: 54000100    	b.eq	0x100000404 <__Z17test_duffs_devicePiPKii+0xa4>
1000003e8: 71001d3f    	cmp	w9, #0x7
1000003ec: 54000080    	b.eq	0x1000003fc <__Z17test_duffs_devicePiPKii+0x9c>
1000003f0: 14000013    	b	0x10000043c <__Z17test_duffs_devicePiPKii+0xdc>
1000003f4: b8404429    	ldr	w9, [x1], #0x4
1000003f8: b8004409    	str	w9, [x0], #0x4
1000003fc: b8404429    	ldr	w9, [x1], #0x4
100000400: b8004409    	str	w9, [x0], #0x4
100000404: b8404429    	ldr	w9, [x1], #0x4
100000408: b8004409    	str	w9, [x0], #0x4
10000040c: b8404429    	ldr	w9, [x1], #0x4
100000410: b8004409    	str	w9, [x0], #0x4
100000414: b8404429    	ldr	w9, [x1], #0x4
100000418: b8004409    	str	w9, [x0], #0x4
10000041c: b8404429    	ldr	w9, [x1], #0x4
100000420: b8004409    	str	w9, [x0], #0x4
100000424: b8404429    	ldr	w9, [x1], #0x4
100000428: b8004409    	str	w9, [x0], #0x4
10000042c: b8404429    	ldr	w9, [x1], #0x4
100000430: b8004409    	str	w9, [x0], #0x4
100000434: 71000508    	subs	w8, w8, #0x1
100000438: 54fffdec    	b.gt	0x1000003f4 <__Z17test_duffs_devicePiPKii+0x94>
10000043c: d65f03c0    	ret

0000000100000440 <_main>:
100000440: 528000a0    	mov	w0, #0x5                ; =5
100000444: d65f03c0    	ret
