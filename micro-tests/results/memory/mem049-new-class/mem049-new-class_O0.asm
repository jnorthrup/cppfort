
/Users/jim/work/cppfort/micro-tests/results/memory/mem049-new-class/mem049-new-class_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000498 <__Z14test_new_classv>:
100000498: d10103ff    	sub	sp, sp, #0x40
10000049c: a9037bfd    	stp	x29, x30, [sp, #0x30]
1000004a0: 9100c3fd    	add	x29, sp, #0x30
1000004a4: d2800080    	mov	x0, #0x4                ; =4
1000004a8: 94000041    	bl	0x1000005ac <___gxx_personality_v0+0x1000005ac>
1000004ac: f9000be0    	str	x0, [sp, #0x10]
1000004b0: 52800541    	mov	w1, #0x2a               ; =42
1000004b4: 9400001b    	bl	0x100000520 <__ZN7CounterC1Ei>
1000004b8: 14000001    	b	0x1000004bc <__Z14test_new_classv+0x24>
1000004bc: f9400be8    	ldr	x8, [sp, #0x10]
1000004c0: f81f83a8    	stur	x8, [x29, #-0x8]
1000004c4: f85f83a0    	ldur	x0, [x29, #-0x8]
1000004c8: 9400003c    	bl	0x1000005b8 <___gxx_personality_v0+0x1000005b8>
1000004cc: b9001be0    	str	w0, [sp, #0x18]
1000004d0: f85f83a8    	ldur	x8, [x29, #-0x8]
1000004d4: f90007e8    	str	x8, [sp, #0x8]
1000004d8: b40000a8    	cbz	x8, 0x1000004ec <__Z14test_new_classv+0x54>
1000004dc: 14000001    	b	0x1000004e0 <__Z14test_new_classv+0x48>
1000004e0: f94007e0    	ldr	x0, [sp, #0x8]
1000004e4: 94000038    	bl	0x1000005c4 <___gxx_personality_v0+0x1000005c4>
1000004e8: 14000001    	b	0x1000004ec <__Z14test_new_classv+0x54>
1000004ec: b9401be0    	ldr	w0, [sp, #0x18]
1000004f0: a9437bfd    	ldp	x29, x30, [sp, #0x30]
1000004f4: 910103ff    	add	sp, sp, #0x40
1000004f8: d65f03c0    	ret
1000004fc: aa0003e8    	mov	x8, x0
100000500: f9400be0    	ldr	x0, [sp, #0x10]
100000504: f81f03a8    	stur	x8, [x29, #-0x10]
100000508: aa0103e8    	mov	x8, x1
10000050c: b81ec3a8    	stur	w8, [x29, #-0x14]
100000510: 9400002d    	bl	0x1000005c4 <___gxx_personality_v0+0x1000005c4>
100000514: 14000001    	b	0x100000518 <__Z14test_new_classv+0x80>
100000518: f85f03a0    	ldur	x0, [x29, #-0x10]
10000051c: 9400002d    	bl	0x1000005d0 <___gxx_personality_v0+0x1000005d0>

0000000100000520 <__ZN7CounterC1Ei>:
100000520: d100c3ff    	sub	sp, sp, #0x30
100000524: a9027bfd    	stp	x29, x30, [sp, #0x20]
100000528: 910083fd    	add	x29, sp, #0x20
10000052c: f81f83a0    	stur	x0, [x29, #-0x8]
100000530: b81f43a1    	stur	w1, [x29, #-0xc]
100000534: f85f83a0    	ldur	x0, [x29, #-0x8]
100000538: f90007e0    	str	x0, [sp, #0x8]
10000053c: b85f43a1    	ldur	w1, [x29, #-0xc]
100000540: 94000013    	bl	0x10000058c <__ZN7CounterC2Ei>
100000544: f94007e0    	ldr	x0, [sp, #0x8]
100000548: a9427bfd    	ldp	x29, x30, [sp, #0x20]
10000054c: 9100c3ff    	add	sp, sp, #0x30
100000550: d65f03c0    	ret

0000000100000554 <__ZN7Counter8getValueEv>:
100000554: d10043ff    	sub	sp, sp, #0x10
100000558: f90007e0    	str	x0, [sp, #0x8]
10000055c: f94007e8    	ldr	x8, [sp, #0x8]
100000560: b9400100    	ldr	w0, [x8]
100000564: 910043ff    	add	sp, sp, #0x10
100000568: d65f03c0    	ret

000000010000056c <_main>:
10000056c: d10083ff    	sub	sp, sp, #0x20
100000570: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000574: 910043fd    	add	x29, sp, #0x10
100000578: b81fc3bf    	stur	wzr, [x29, #-0x4]
10000057c: 97ffffc7    	bl	0x100000498 <__Z14test_new_classv>
100000580: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000584: 910083ff    	add	sp, sp, #0x20
100000588: d65f03c0    	ret

000000010000058c <__ZN7CounterC2Ei>:
10000058c: d10043ff    	sub	sp, sp, #0x10
100000590: f90007e0    	str	x0, [sp, #0x8]
100000594: b90007e1    	str	w1, [sp, #0x4]
100000598: f94007e0    	ldr	x0, [sp, #0x8]
10000059c: b94007e8    	ldr	w8, [sp, #0x4]
1000005a0: b9000008    	str	w8, [x0]
1000005a4: 910043ff    	add	sp, sp, #0x10
1000005a8: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

00000001000005ac <__stubs>:
1000005ac: 90000030    	adrp	x16, 0x100004000 <___gxx_personality_v0+0x100004000>
1000005b0: f9400610    	ldr	x16, [x16, #0x8]
1000005b4: d61f0200    	br	x16
1000005b8: 90000030    	adrp	x16, 0x100004000 <___gxx_personality_v0+0x100004000>
1000005bc: f9400a10    	ldr	x16, [x16, #0x10]
1000005c0: d61f0200    	br	x16
1000005c4: 90000030    	adrp	x16, 0x100004000 <___gxx_personality_v0+0x100004000>
1000005c8: f9400e10    	ldr	x16, [x16, #0x18]
1000005cc: d61f0200    	br	x16
1000005d0: 90000030    	adrp	x16, 0x100004000 <___gxx_personality_v0+0x100004000>
1000005d4: f9401210    	ldr	x16, [x16, #0x20]
1000005d8: d61f0200    	br	x16
