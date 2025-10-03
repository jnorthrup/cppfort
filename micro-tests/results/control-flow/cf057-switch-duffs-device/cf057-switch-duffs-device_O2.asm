
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf057-switch-duffs-device/cf057-switch-duffs-device_O2.out:	file format mach-o arm64

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
100000384: 540000ec    	b.gt	0x1000003a0 <__Z17test_duffs_devicePiPKii+0x40>
100000388: 7100053f    	cmp	w9, #0x1
10000038c: 5400018c    	b.gt	0x1000003bc <__Z17test_duffs_devicePiPKii+0x5c>
100000390: 340002a9    	cbz	w9, 0x1000003e4 <__Z17test_duffs_devicePiPKii+0x84>
100000394: 7100053f    	cmp	w9, #0x1
100000398: 54000420    	b.eq	0x10000041c <__Z17test_duffs_devicePiPKii+0xbc>
10000039c: 14000024    	b	0x10000042c <__Z17test_duffs_devicePiPKii+0xcc>
1000003a0: 7100153f    	cmp	w9, #0x5
1000003a4: 5400016c    	b.gt	0x1000003d0 <__Z17test_duffs_devicePiPKii+0x70>
1000003a8: 7100113f    	cmp	w9, #0x4
1000003ac: 540002c0    	b.eq	0x100000404 <__Z17test_duffs_devicePiPKii+0xa4>
1000003b0: 7100153f    	cmp	w9, #0x5
1000003b4: 54000240    	b.eq	0x1000003fc <__Z17test_duffs_devicePiPKii+0x9c>
1000003b8: 1400001d    	b	0x10000042c <__Z17test_duffs_devicePiPKii+0xcc>
1000003bc: 7100093f    	cmp	w9, #0x2
1000003c0: 540002a0    	b.eq	0x100000414 <__Z17test_duffs_devicePiPKii+0xb4>
1000003c4: 71000d3f    	cmp	w9, #0x3
1000003c8: 54000220    	b.eq	0x10000040c <__Z17test_duffs_devicePiPKii+0xac>
1000003cc: 14000018    	b	0x10000042c <__Z17test_duffs_devicePiPKii+0xcc>
1000003d0: 7100193f    	cmp	w9, #0x6
1000003d4: 54000100    	b.eq	0x1000003f4 <__Z17test_duffs_devicePiPKii+0x94>
1000003d8: 71001d3f    	cmp	w9, #0x7
1000003dc: 54000080    	b.eq	0x1000003ec <__Z17test_duffs_devicePiPKii+0x8c>
1000003e0: 14000013    	b	0x10000042c <__Z17test_duffs_devicePiPKii+0xcc>
1000003e4: b8404429    	ldr	w9, [x1], #0x4
1000003e8: b8004409    	str	w9, [x0], #0x4
1000003ec: b8404429    	ldr	w9, [x1], #0x4
1000003f0: b8004409    	str	w9, [x0], #0x4
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
100000424: 71000508    	subs	w8, w8, #0x1
100000428: 54fffdec    	b.gt	0x1000003e4 <__Z17test_duffs_devicePiPKii+0x84>
10000042c: d65f03c0    	ret

0000000100000430 <_main>:
100000430: 528000a0    	mov	w0, #0x5                ; =5
100000434: d65f03c0    	ret
