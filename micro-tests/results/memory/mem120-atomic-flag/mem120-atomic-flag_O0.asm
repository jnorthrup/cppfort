
/Users/jim/work/cppfort/micro-tests/results/memory/mem120-atomic-flag/mem120-atomic-flag_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

00000001000003f8 <__Z16test_atomic_flagv>:
1000003f8: d100c3ff    	sub	sp, sp, #0x30
1000003fc: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000400: 910083fd    	add	x29, sp, #0x20
100000404: 90000020    	adrp	x0, 0x100004000 <_flag>
100000408: 91000000    	add	x0, x0, #0x0
10000040c: f9000be0    	str	x0, [sp, #0x10]
100000410: 528000a1    	mov	w1, #0x5                ; =5
100000414: b9000fe1    	str	w1, [sp, #0xc]
100000418: 9400000e    	bl	0x100000450 <__ZNSt3__111atomic_flag12test_and_setB8ne200100ENS_12memory_orderE>
10000041c: b9400fe1    	ldr	w1, [sp, #0xc]
100000420: aa0003e8    	mov	x8, x0
100000424: f9400be0    	ldr	x0, [sp, #0x10]
100000428: 381ff3a8    	sturb	w8, [x29, #-0x1]
10000042c: 94000016    	bl	0x100000484 <__ZNSt3__111atomic_flag5clearB8ne200100ENS_12memory_orderE>
100000430: 385ff3a9    	ldurb	w9, [x29, #-0x1]
100000434: 52800008    	mov	w8, #0x0                ; =0
100000438: 12000129    	and	w9, w9, #0x1
10000043c: 72000129    	ands	w9, w9, #0x1
100000440: 1a9f0500    	csinc	w0, w8, wzr, eq
100000444: a9427bfd    	ldp	x29, x30, [sp, #0x20]
100000448: 9100c3ff    	add	sp, sp, #0x30
10000044c: d65f03c0    	ret

0000000100000450 <__ZNSt3__111atomic_flag12test_and_setB8ne200100ENS_12memory_orderE>:
100000450: d10083ff    	sub	sp, sp, #0x20
100000454: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000458: 910043fd    	add	x29, sp, #0x10
10000045c: f90007e0    	str	x0, [sp, #0x8]
100000460: b90007e1    	str	w1, [sp, #0x4]
100000464: f94007e0    	ldr	x0, [sp, #0x8]
100000468: b94007e2    	ldr	w2, [sp, #0x4]
10000046c: 52800028    	mov	w8, #0x1                ; =1
100000470: 12000101    	and	w1, w8, #0x1
100000474: 94000019    	bl	0x1000004d8 <__ZNSt3__121__cxx_atomic_exchangeB8ne200100IbEET_PNS_22__cxx_atomic_base_implIS1_EES1_NS_12memory_orderE>
100000478: a9417bfd    	ldp	x29, x30, [sp, #0x10]
10000047c: 910083ff    	add	sp, sp, #0x20
100000480: d65f03c0    	ret

0000000100000484 <__ZNSt3__111atomic_flag5clearB8ne200100ENS_12memory_orderE>:
100000484: d10083ff    	sub	sp, sp, #0x20
100000488: a9017bfd    	stp	x29, x30, [sp, #0x10]
10000048c: 910043fd    	add	x29, sp, #0x10
100000490: f90007e0    	str	x0, [sp, #0x8]
100000494: b90007e1    	str	w1, [sp, #0x4]
100000498: f94007e0    	ldr	x0, [sp, #0x8]
10000049c: b94007e2    	ldr	w2, [sp, #0x4]
1000004a0: 52800008    	mov	w8, #0x0                ; =0
1000004a4: 12000101    	and	w1, w8, #0x1
1000004a8: 94000045    	bl	0x1000005bc <__ZNSt3__118__cxx_atomic_storeB8ne200100IbEEvPNS_22__cxx_atomic_base_implIT_EES2_NS_12memory_orderE>
1000004ac: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000004b0: 910083ff    	add	sp, sp, #0x20
1000004b4: d65f03c0    	ret

00000001000004b8 <_main>:
1000004b8: d10083ff    	sub	sp, sp, #0x20
1000004bc: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000004c0: 910043fd    	add	x29, sp, #0x10
1000004c4: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000004c8: 97ffffcc    	bl	0x1000003f8 <__Z16test_atomic_flagv>
1000004cc: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000004d0: 910083ff    	add	sp, sp, #0x20
1000004d4: d65f03c0    	ret

00000001000004d8 <__ZNSt3__121__cxx_atomic_exchangeB8ne200100IbEET_PNS_22__cxx_atomic_base_implIS1_EES1_NS_12memory_orderE>:
1000004d8: d10083ff    	sub	sp, sp, #0x20
1000004dc: f9000fe0    	str	x0, [sp, #0x18]
1000004e0: 39005fe1    	strb	w1, [sp, #0x17]
1000004e4: b90013e2    	str	w2, [sp, #0x10]
1000004e8: f9400fe8    	ldr	x8, [sp, #0x18]
1000004ec: f90003e8    	str	x8, [sp]
1000004f0: b94013e8    	ldr	w8, [sp, #0x10]
1000004f4: b9000be8    	str	w8, [sp, #0x8]
1000004f8: 39405fe9    	ldrb	w9, [sp, #0x17]
1000004fc: 5280002a    	mov	w10, #0x1               ; =1
100000500: 0a0a0129    	and	w9, w9, w10
100000504: 39003fe9    	strb	w9, [sp, #0xf]
100000508: 71000508    	subs	w8, w8, #0x1
10000050c: 71000508    	subs	w8, w8, #0x1
100000510: 54000269    	b.ls	0x10000055c <__ZNSt3__121__cxx_atomic_exchangeB8ne200100IbEET_PNS_22__cxx_atomic_base_implIS1_EES1_NS_12memory_orderE+0x84>
100000514: 14000001    	b	0x100000518 <__ZNSt3__121__cxx_atomic_exchangeB8ne200100IbEET_PNS_22__cxx_atomic_base_implIS1_EES1_NS_12memory_orderE+0x40>
100000518: b9400be8    	ldr	w8, [sp, #0x8]
10000051c: 71000d08    	subs	w8, w8, #0x3
100000520: 54000280    	b.eq	0x100000570 <__ZNSt3__121__cxx_atomic_exchangeB8ne200100IbEET_PNS_22__cxx_atomic_base_implIS1_EES1_NS_12memory_orderE+0x98>
100000524: 14000001    	b	0x100000528 <__ZNSt3__121__cxx_atomic_exchangeB8ne200100IbEET_PNS_22__cxx_atomic_base_implIS1_EES1_NS_12memory_orderE+0x50>
100000528: b9400be8    	ldr	w8, [sp, #0x8]
10000052c: 71001108    	subs	w8, w8, #0x4
100000530: 540002a0    	b.eq	0x100000584 <__ZNSt3__121__cxx_atomic_exchangeB8ne200100IbEET_PNS_22__cxx_atomic_base_implIS1_EES1_NS_12memory_orderE+0xac>
100000534: 14000001    	b	0x100000538 <__ZNSt3__121__cxx_atomic_exchangeB8ne200100IbEET_PNS_22__cxx_atomic_base_implIS1_EES1_NS_12memory_orderE+0x60>
100000538: b9400be8    	ldr	w8, [sp, #0x8]
10000053c: 71001508    	subs	w8, w8, #0x5
100000540: 540002c0    	b.eq	0x100000598 <__ZNSt3__121__cxx_atomic_exchangeB8ne200100IbEET_PNS_22__cxx_atomic_base_implIS1_EES1_NS_12memory_orderE+0xc0>
100000544: 14000001    	b	0x100000548 <__ZNSt3__121__cxx_atomic_exchangeB8ne200100IbEET_PNS_22__cxx_atomic_base_implIS1_EES1_NS_12memory_orderE+0x70>
100000548: f94003e9    	ldr	x9, [sp]
10000054c: 39403fe8    	ldrb	w8, [sp, #0xf]
100000550: 38288128    	swpb	w8, w8, [x9]
100000554: 39003be8    	strb	w8, [sp, #0xe]
100000558: 14000015    	b	0x1000005ac <__ZNSt3__121__cxx_atomic_exchangeB8ne200100IbEET_PNS_22__cxx_atomic_base_implIS1_EES1_NS_12memory_orderE+0xd4>
10000055c: f94003e9    	ldr	x9, [sp]
100000560: 39403fe8    	ldrb	w8, [sp, #0xf]
100000564: 38a88128    	swpab	w8, w8, [x9]
100000568: 39003be8    	strb	w8, [sp, #0xe]
10000056c: 14000010    	b	0x1000005ac <__ZNSt3__121__cxx_atomic_exchangeB8ne200100IbEET_PNS_22__cxx_atomic_base_implIS1_EES1_NS_12memory_orderE+0xd4>
100000570: f94003e9    	ldr	x9, [sp]
100000574: 39403fe8    	ldrb	w8, [sp, #0xf]
100000578: 38688128    	swplb	w8, w8, [x9]
10000057c: 39003be8    	strb	w8, [sp, #0xe]
100000580: 1400000b    	b	0x1000005ac <__ZNSt3__121__cxx_atomic_exchangeB8ne200100IbEET_PNS_22__cxx_atomic_base_implIS1_EES1_NS_12memory_orderE+0xd4>
100000584: f94003e9    	ldr	x9, [sp]
100000588: 39403fe8    	ldrb	w8, [sp, #0xf]
10000058c: 38e88128    	swpalb	w8, w8, [x9]
100000590: 39003be8    	strb	w8, [sp, #0xe]
100000594: 14000006    	b	0x1000005ac <__ZNSt3__121__cxx_atomic_exchangeB8ne200100IbEET_PNS_22__cxx_atomic_base_implIS1_EES1_NS_12memory_orderE+0xd4>
100000598: f94003e9    	ldr	x9, [sp]
10000059c: 39403fe8    	ldrb	w8, [sp, #0xf]
1000005a0: 38e88128    	swpalb	w8, w8, [x9]
1000005a4: 39003be8    	strb	w8, [sp, #0xe]
1000005a8: 14000001    	b	0x1000005ac <__ZNSt3__121__cxx_atomic_exchangeB8ne200100IbEET_PNS_22__cxx_atomic_base_implIS1_EES1_NS_12memory_orderE+0xd4>
1000005ac: 39403be8    	ldrb	w8, [sp, #0xe]
1000005b0: 12000100    	and	w0, w8, #0x1
1000005b4: 910083ff    	add	sp, sp, #0x20
1000005b8: d65f03c0    	ret

00000001000005bc <__ZNSt3__118__cxx_atomic_storeB8ne200100IbEEvPNS_22__cxx_atomic_base_implIT_EES2_NS_12memory_orderE>:
1000005bc: d10083ff    	sub	sp, sp, #0x20
1000005c0: f9000fe0    	str	x0, [sp, #0x18]
1000005c4: 39005fe1    	strb	w1, [sp, #0x17]
1000005c8: b90013e2    	str	w2, [sp, #0x10]
1000005cc: f9400fe8    	ldr	x8, [sp, #0x18]
1000005d0: f90003e8    	str	x8, [sp]
1000005d4: b94013e8    	ldr	w8, [sp, #0x10]
1000005d8: b9000be8    	str	w8, [sp, #0x8]
1000005dc: 39405fe9    	ldrb	w9, [sp, #0x17]
1000005e0: 12000129    	and	w9, w9, #0x1
1000005e4: 39003fe9    	strb	w9, [sp, #0xf]
1000005e8: 71000d08    	subs	w8, w8, #0x3
1000005ec: 54000140    	b.eq	0x100000614 <__ZNSt3__118__cxx_atomic_storeB8ne200100IbEEvPNS_22__cxx_atomic_base_implIT_EES2_NS_12memory_orderE+0x58>
1000005f0: 14000001    	b	0x1000005f4 <__ZNSt3__118__cxx_atomic_storeB8ne200100IbEEvPNS_22__cxx_atomic_base_implIT_EES2_NS_12memory_orderE+0x38>
1000005f4: b9400be8    	ldr	w8, [sp, #0x8]
1000005f8: 71001508    	subs	w8, w8, #0x5
1000005fc: 54000140    	b.eq	0x100000624 <__ZNSt3__118__cxx_atomic_storeB8ne200100IbEEvPNS_22__cxx_atomic_base_implIT_EES2_NS_12memory_orderE+0x68>
100000600: 14000001    	b	0x100000604 <__ZNSt3__118__cxx_atomic_storeB8ne200100IbEEvPNS_22__cxx_atomic_base_implIT_EES2_NS_12memory_orderE+0x48>
100000604: f94003e9    	ldr	x9, [sp]
100000608: 39403fe8    	ldrb	w8, [sp, #0xf]
10000060c: 39000128    	strb	w8, [x9]
100000610: 14000009    	b	0x100000634 <__ZNSt3__118__cxx_atomic_storeB8ne200100IbEEvPNS_22__cxx_atomic_base_implIT_EES2_NS_12memory_orderE+0x78>
100000614: f94003e9    	ldr	x9, [sp]
100000618: 39403fe8    	ldrb	w8, [sp, #0xf]
10000061c: 089ffd28    	stlrb	w8, [x9]
100000620: 14000005    	b	0x100000634 <__ZNSt3__118__cxx_atomic_storeB8ne200100IbEEvPNS_22__cxx_atomic_base_implIT_EES2_NS_12memory_orderE+0x78>
100000624: f94003e9    	ldr	x9, [sp]
100000628: 39403fe8    	ldrb	w8, [sp, #0xf]
10000062c: 089ffd28    	stlrb	w8, [x9]
100000630: 14000001    	b	0x100000634 <__ZNSt3__118__cxx_atomic_storeB8ne200100IbEEvPNS_22__cxx_atomic_base_implIT_EES2_NS_12memory_orderE+0x78>
100000634: 910083ff    	add	sp, sp, #0x20
100000638: d65f03c0    	ret
