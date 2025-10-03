
/Users/jim/work/cppfort/micro-tests/results/classes/cls074-ctor/cls074-ctor_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000498 <_main>:
100000498: d100c3ff    	sub	sp, sp, #0x30
10000049c: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000004a0: 910083fd    	add	x29, sp, #0x20
1000004a4: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000004a8: d10023a0    	sub	x0, x29, #0x8
1000004ac: f90003e0    	str	x0, [sp]
1000004b0: 94000015    	bl	0x100000504 <__ZN4TestC1Ev>
1000004b4: f94003e0    	ldr	x0, [sp]
1000004b8: 9400003b    	bl	0x1000005a4 <___gxx_personality_v0+0x1000005a4>
1000004bc: b9000be0    	str	w0, [sp, #0x8]
1000004c0: 14000001    	b	0x1000004c4 <_main+0x2c>
1000004c4: b9400be8    	ldr	w8, [sp, #0x8]
1000004c8: b81fc3a8    	stur	w8, [x29, #-0x4]
1000004cc: d10023a0    	sub	x0, x29, #0x8
1000004d0: 9400001e    	bl	0x100000548 <__ZN4TestD1Ev>
1000004d4: b85fc3a0    	ldur	w0, [x29, #-0x4]
1000004d8: a9427bfd    	ldp	x29, x30, [sp, #0x20]
1000004dc: 9100c3ff    	add	sp, sp, #0x30
1000004e0: d65f03c0    	ret
1000004e4: f9000be0    	str	x0, [sp, #0x10]
1000004e8: aa0103e8    	mov	x8, x1
1000004ec: b9000fe8    	str	w8, [sp, #0xc]
1000004f0: d10023a0    	sub	x0, x29, #0x8
1000004f4: 94000015    	bl	0x100000548 <__ZN4TestD1Ev>
1000004f8: 14000001    	b	0x1000004fc <_main+0x64>
1000004fc: f9400be0    	ldr	x0, [sp, #0x10]
100000500: 9400002c    	bl	0x1000005b0 <___gxx_personality_v0+0x1000005b0>

0000000100000504 <__ZN4TestC1Ev>:
100000504: d10083ff    	sub	sp, sp, #0x20
100000508: a9017bfd    	stp	x29, x30, [sp, #0x10]
10000050c: 910043fd    	add	x29, sp, #0x10
100000510: f90007e0    	str	x0, [sp, #0x8]
100000514: f94007e0    	ldr	x0, [sp, #0x8]
100000518: f90003e0    	str	x0, [sp]
10000051c: 94000016    	bl	0x100000574 <__ZN4TestC2Ev>
100000520: f94003e0    	ldr	x0, [sp]
100000524: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000528: 910083ff    	add	sp, sp, #0x20
10000052c: d65f03c0    	ret

0000000100000530 <__ZN4Test3getEv>:
100000530: d10043ff    	sub	sp, sp, #0x10
100000534: f90007e0    	str	x0, [sp, #0x8]
100000538: f94007e8    	ldr	x8, [sp, #0x8]
10000053c: b9400100    	ldr	w0, [x8]
100000540: 910043ff    	add	sp, sp, #0x10
100000544: d65f03c0    	ret

0000000100000548 <__ZN4TestD1Ev>:
100000548: d10083ff    	sub	sp, sp, #0x20
10000054c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000550: 910043fd    	add	x29, sp, #0x10
100000554: f90007e0    	str	x0, [sp, #0x8]
100000558: f94007e0    	ldr	x0, [sp, #0x8]
10000055c: f90003e0    	str	x0, [sp]
100000560: 9400000c    	bl	0x100000590 <__ZN4TestD2Ev>
100000564: f94003e0    	ldr	x0, [sp]
100000568: a9417bfd    	ldp	x29, x30, [sp, #0x10]
10000056c: 910083ff    	add	sp, sp, #0x20
100000570: d65f03c0    	ret

0000000100000574 <__ZN4TestC2Ev>:
100000574: d10043ff    	sub	sp, sp, #0x10
100000578: f90007e0    	str	x0, [sp, #0x8]
10000057c: f94007e0    	ldr	x0, [sp, #0x8]
100000580: 52800948    	mov	w8, #0x4a               ; =74
100000584: b9000008    	str	w8, [x0]
100000588: 910043ff    	add	sp, sp, #0x10
10000058c: d65f03c0    	ret

0000000100000590 <__ZN4TestD2Ev>:
100000590: d10043ff    	sub	sp, sp, #0x10
100000594: f90007e0    	str	x0, [sp, #0x8]
100000598: f94007e0    	ldr	x0, [sp, #0x8]
10000059c: 910043ff    	add	sp, sp, #0x10
1000005a0: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

00000001000005a4 <__stubs>:
1000005a4: 90000030    	adrp	x16, 0x100004000 <___gxx_personality_v0+0x100004000>
1000005a8: f9400610    	ldr	x16, [x16, #0x8]
1000005ac: d61f0200    	br	x16
1000005b0: 90000030    	adrp	x16, 0x100004000 <___gxx_personality_v0+0x100004000>
1000005b4: f9400a10    	ldr	x16, [x16, #0x10]
1000005b8: d61f0200    	br	x16
