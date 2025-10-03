
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf057-switch-duffs-device/cf057-switch-duffs-device_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000498 <__Z17test_duffs_devicePiPKii>:
100000498: d10083ff    	sub	sp, sp, #0x20
10000049c: f9000fe0    	str	x0, [sp, #0x18]
1000004a0: f9000be1    	str	x1, [sp, #0x10]
1000004a4: b9000fe2    	str	w2, [sp, #0xc]
1000004a8: b9400fe8    	ldr	w8, [sp, #0xc]
1000004ac: 11001d08    	add	w8, w8, #0x7
1000004b0: 5280010a    	mov	w10, #0x8               ; =8
1000004b4: 1aca0d08    	sdiv	w8, w8, w10
1000004b8: b9000be8    	str	w8, [sp, #0x8]
1000004bc: b9400fe8    	ldr	w8, [sp, #0xc]
1000004c0: 1aca0d09    	sdiv	w9, w8, w10
1000004c4: 1b0a7d29    	mul	w9, w9, w10
1000004c8: 6b090108    	subs	w8, w8, w9
1000004cc: b90007e8    	str	w8, [sp, #0x4]
1000004d0: 340003c8    	cbz	w8, 0x100000548 <__Z17test_duffs_devicePiPKii+0xb0>
1000004d4: 14000001    	b	0x1000004d8 <__Z17test_duffs_devicePiPKii+0x40>
1000004d8: b94007e8    	ldr	w8, [sp, #0x4]
1000004dc: 71000508    	subs	w8, w8, #0x1
1000004e0: 54000b40    	b.eq	0x100000648 <__Z17test_duffs_devicePiPKii+0x1b0>
1000004e4: 14000001    	b	0x1000004e8 <__Z17test_duffs_devicePiPKii+0x50>
1000004e8: b94007e8    	ldr	w8, [sp, #0x4]
1000004ec: 71000908    	subs	w8, w8, #0x2
1000004f0: 540009a0    	b.eq	0x100000624 <__Z17test_duffs_devicePiPKii+0x18c>
1000004f4: 14000001    	b	0x1000004f8 <__Z17test_duffs_devicePiPKii+0x60>
1000004f8: b94007e8    	ldr	w8, [sp, #0x4]
1000004fc: 71000d08    	subs	w8, w8, #0x3
100000500: 54000800    	b.eq	0x100000600 <__Z17test_duffs_devicePiPKii+0x168>
100000504: 14000001    	b	0x100000508 <__Z17test_duffs_devicePiPKii+0x70>
100000508: b94007e8    	ldr	w8, [sp, #0x4]
10000050c: 71001108    	subs	w8, w8, #0x4
100000510: 54000660    	b.eq	0x1000005dc <__Z17test_duffs_devicePiPKii+0x144>
100000514: 14000001    	b	0x100000518 <__Z17test_duffs_devicePiPKii+0x80>
100000518: b94007e8    	ldr	w8, [sp, #0x4]
10000051c: 71001508    	subs	w8, w8, #0x5
100000520: 540004c0    	b.eq	0x1000005b8 <__Z17test_duffs_devicePiPKii+0x120>
100000524: 14000001    	b	0x100000528 <__Z17test_duffs_devicePiPKii+0x90>
100000528: b94007e8    	ldr	w8, [sp, #0x4]
10000052c: 71001908    	subs	w8, w8, #0x6
100000530: 54000320    	b.eq	0x100000594 <__Z17test_duffs_devicePiPKii+0xfc>
100000534: 14000001    	b	0x100000538 <__Z17test_duffs_devicePiPKii+0xa0>
100000538: b94007e8    	ldr	w8, [sp, #0x4]
10000053c: 71001d08    	subs	w8, w8, #0x7
100000540: 54000180    	b.eq	0x100000570 <__Z17test_duffs_devicePiPKii+0xd8>
100000544: 14000051    	b	0x100000688 <__Z17test_duffs_devicePiPKii+0x1f0>
100000548: 14000001    	b	0x10000054c <__Z17test_duffs_devicePiPKii+0xb4>
10000054c: f9400be8    	ldr	x8, [sp, #0x10]
100000550: 91001109    	add	x9, x8, #0x4
100000554: f9000be9    	str	x9, [sp, #0x10]
100000558: b9400108    	ldr	w8, [x8]
10000055c: f9400fe9    	ldr	x9, [sp, #0x18]
100000560: 9100112a    	add	x10, x9, #0x4
100000564: f9000fea    	str	x10, [sp, #0x18]
100000568: b9000128    	str	w8, [x9]
10000056c: 14000001    	b	0x100000570 <__Z17test_duffs_devicePiPKii+0xd8>
100000570: f9400be8    	ldr	x8, [sp, #0x10]
100000574: 91001109    	add	x9, x8, #0x4
100000578: f9000be9    	str	x9, [sp, #0x10]
10000057c: b9400108    	ldr	w8, [x8]
100000580: f9400fe9    	ldr	x9, [sp, #0x18]
100000584: 9100112a    	add	x10, x9, #0x4
100000588: f9000fea    	str	x10, [sp, #0x18]
10000058c: b9000128    	str	w8, [x9]
100000590: 14000001    	b	0x100000594 <__Z17test_duffs_devicePiPKii+0xfc>
100000594: f9400be8    	ldr	x8, [sp, #0x10]
100000598: 91001109    	add	x9, x8, #0x4
10000059c: f9000be9    	str	x9, [sp, #0x10]
1000005a0: b9400108    	ldr	w8, [x8]
1000005a4: f9400fe9    	ldr	x9, [sp, #0x18]
1000005a8: 9100112a    	add	x10, x9, #0x4
1000005ac: f9000fea    	str	x10, [sp, #0x18]
1000005b0: b9000128    	str	w8, [x9]
1000005b4: 14000001    	b	0x1000005b8 <__Z17test_duffs_devicePiPKii+0x120>
1000005b8: f9400be8    	ldr	x8, [sp, #0x10]
1000005bc: 91001109    	add	x9, x8, #0x4
1000005c0: f9000be9    	str	x9, [sp, #0x10]
1000005c4: b9400108    	ldr	w8, [x8]
1000005c8: f9400fe9    	ldr	x9, [sp, #0x18]
1000005cc: 9100112a    	add	x10, x9, #0x4
1000005d0: f9000fea    	str	x10, [sp, #0x18]
1000005d4: b9000128    	str	w8, [x9]
1000005d8: 14000001    	b	0x1000005dc <__Z17test_duffs_devicePiPKii+0x144>
1000005dc: f9400be8    	ldr	x8, [sp, #0x10]
1000005e0: 91001109    	add	x9, x8, #0x4
1000005e4: f9000be9    	str	x9, [sp, #0x10]
1000005e8: b9400108    	ldr	w8, [x8]
1000005ec: f9400fe9    	ldr	x9, [sp, #0x18]
1000005f0: 9100112a    	add	x10, x9, #0x4
1000005f4: f9000fea    	str	x10, [sp, #0x18]
1000005f8: b9000128    	str	w8, [x9]
1000005fc: 14000001    	b	0x100000600 <__Z17test_duffs_devicePiPKii+0x168>
100000600: f9400be8    	ldr	x8, [sp, #0x10]
100000604: 91001109    	add	x9, x8, #0x4
100000608: f9000be9    	str	x9, [sp, #0x10]
10000060c: b9400108    	ldr	w8, [x8]
100000610: f9400fe9    	ldr	x9, [sp, #0x18]
100000614: 9100112a    	add	x10, x9, #0x4
100000618: f9000fea    	str	x10, [sp, #0x18]
10000061c: b9000128    	str	w8, [x9]
100000620: 14000001    	b	0x100000624 <__Z17test_duffs_devicePiPKii+0x18c>
100000624: f9400be8    	ldr	x8, [sp, #0x10]
100000628: 91001109    	add	x9, x8, #0x4
10000062c: f9000be9    	str	x9, [sp, #0x10]
100000630: b9400108    	ldr	w8, [x8]
100000634: f9400fe9    	ldr	x9, [sp, #0x18]
100000638: 9100112a    	add	x10, x9, #0x4
10000063c: f9000fea    	str	x10, [sp, #0x18]
100000640: b9000128    	str	w8, [x9]
100000644: 14000001    	b	0x100000648 <__Z17test_duffs_devicePiPKii+0x1b0>
100000648: f9400be8    	ldr	x8, [sp, #0x10]
10000064c: 91001109    	add	x9, x8, #0x4
100000650: f9000be9    	str	x9, [sp, #0x10]
100000654: b9400108    	ldr	w8, [x8]
100000658: f9400fe9    	ldr	x9, [sp, #0x18]
10000065c: 9100112a    	add	x10, x9, #0x4
100000660: f9000fea    	str	x10, [sp, #0x18]
100000664: b9000128    	str	w8, [x9]
100000668: 14000001    	b	0x10000066c <__Z17test_duffs_devicePiPKii+0x1d4>
10000066c: b9400be8    	ldr	w8, [sp, #0x8]
100000670: 71000508    	subs	w8, w8, #0x1
100000674: b9000be8    	str	w8, [sp, #0x8]
100000678: 71000108    	subs	w8, w8, #0x0
10000067c: 54fff68c    	b.gt	0x10000054c <__Z17test_duffs_devicePiPKii+0xb4>
100000680: 14000001    	b	0x100000684 <__Z17test_duffs_devicePiPKii+0x1ec>
100000684: 14000001    	b	0x100000688 <__Z17test_duffs_devicePiPKii+0x1f0>
100000688: 910083ff    	add	sp, sp, #0x20
10000068c: d65f03c0    	ret

0000000100000690 <_main>:
100000690: d10243ff    	sub	sp, sp, #0x90
100000694: a9087bfd    	stp	x29, x30, [sp, #0x80]
100000698: 910203fd    	add	x29, sp, #0x80
10000069c: 90000028    	adrp	x8, 0x100004000 <_memset+0x100004000>
1000006a0: f9400108    	ldr	x8, [x8]
1000006a4: f9400108    	ldr	x8, [x8]
1000006a8: f81f83a8    	stur	x8, [x29, #-0x8]
1000006ac: 52800008    	mov	w8, #0x0                ; =0
1000006b0: b90007e8    	str	w8, [sp, #0x4]
1000006b4: b90027ff    	str	wzr, [sp, #0x24]
1000006b8: d100c3a0    	sub	x0, x29, #0x30
1000006bc: f9000fe0    	str	x0, [sp, #0x18]
1000006c0: d2800502    	mov	x2, #0x28               ; =40
1000006c4: f90007e2    	str	x2, [sp, #0x8]
1000006c8: 90000001    	adrp	x1, 0x100000000 <_memset+0x100000000>
1000006cc: 911d5021    	add	x1, x1, #0x754
1000006d0: 94000018    	bl	0x100000730 <_memset+0x100000730>
1000006d4: b94007e1    	ldr	w1, [sp, #0x4]
1000006d8: f94007e2    	ldr	x2, [sp, #0x8]
1000006dc: 9100a3e0    	add	x0, sp, #0x28
1000006e0: f9000be0    	str	x0, [sp, #0x10]
1000006e4: 94000016    	bl	0x10000073c <_memset+0x10000073c>
1000006e8: f9400be0    	ldr	x0, [sp, #0x10]
1000006ec: f9400fe1    	ldr	x1, [sp, #0x18]
1000006f0: 52800142    	mov	w2, #0xa                ; =10
1000006f4: 97ffff69    	bl	0x100000498 <__Z17test_duffs_devicePiPKii>
1000006f8: b9403fe8    	ldr	w8, [sp, #0x3c]
1000006fc: b90023e8    	str	w8, [sp, #0x20]
100000700: f85f83a9    	ldur	x9, [x29, #-0x8]
100000704: 90000028    	adrp	x8, 0x100004000 <_memset+0x100004000>
100000708: f9400108    	ldr	x8, [x8]
10000070c: f9400108    	ldr	x8, [x8]
100000710: eb090108    	subs	x8, x8, x9
100000714: 54000060    	b.eq	0x100000720 <_main+0x90>
100000718: 14000001    	b	0x10000071c <_main+0x8c>
10000071c: 9400000b    	bl	0x100000748 <_memset+0x100000748>
100000720: b94023e0    	ldr	w0, [sp, #0x20]
100000724: a9487bfd    	ldp	x29, x30, [sp, #0x80]
100000728: 910243ff    	add	sp, sp, #0x90
10000072c: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

0000000100000730 <__stubs>:
100000730: 90000030    	adrp	x16, 0x100004000 <_memset+0x100004000>
100000734: f9400610    	ldr	x16, [x16, #0x8]
100000738: d61f0200    	br	x16
10000073c: 90000030    	adrp	x16, 0x100004000 <_memset+0x100004000>
100000740: f9400a10    	ldr	x16, [x16, #0x10]
100000744: d61f0200    	br	x16
100000748: 90000030    	adrp	x16, 0x100004000 <_memset+0x100004000>
10000074c: f9400e10    	ldr	x16, [x16, #0x18]
100000750: d61f0200    	br	x16
