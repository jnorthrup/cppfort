
/Users/jim/work/cppfort/micro-tests/results/memory/mem096-jagged-array/mem096-jagged-array_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000498 <__Z17test_jagged_arrayv>:
100000498: d10143ff    	sub	sp, sp, #0x50
10000049c: a9047bfd    	stp	x29, x30, [sp, #0x40]
1000004a0: 910103fd    	add	x29, sp, #0x40
1000004a4: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
1000004a8: f9400108    	ldr	x8, [x8]
1000004ac: f9400108    	ldr	x8, [x8]
1000004b0: f81f83a8    	stur	x8, [x29, #-0x8]
1000004b4: 90000008    	adrp	x8, 0x100000000 <___stack_chk_guard+0x100000000>
1000004b8: 9115a108    	add	x8, x8, #0x568
1000004bc: f9400108    	ldr	x8, [x8]
1000004c0: 910063ea    	add	x10, sp, #0x18
1000004c4: f9000fe8    	str	x8, [sp, #0x18]
1000004c8: 90000008    	adrp	x8, 0x100000000 <___stack_chk_guard+0x100000000>
1000004cc: 9115c108    	add	x8, x8, #0x570
1000004d0: f940010b    	ldr	x11, [x8]
1000004d4: 910023e9    	add	x9, sp, #0x8
1000004d8: f90007eb    	str	x11, [sp, #0x8]
1000004dc: b9400908    	ldr	w8, [x8, #0x8]
1000004e0: b90013e8    	str	w8, [sp, #0x10]
1000004e4: 90000008    	adrp	x8, 0x100000000 <___stack_chk_guard+0x100000000>
1000004e8: b9457d0b    	ldr	w11, [x8, #0x57c]
1000004ec: 910013e8    	add	x8, sp, #0x4
1000004f0: b90007eb    	str	w11, [sp, #0x4]
1000004f4: f90013ea    	str	x10, [sp, #0x20]
1000004f8: f90017e9    	str	x9, [sp, #0x28]
1000004fc: f9001be8    	str	x8, [sp, #0x30]
100000500: f94017e8    	ldr	x8, [sp, #0x28]
100000504: b9400508    	ldr	w8, [x8, #0x4]
100000508: b90003e8    	str	w8, [sp]
10000050c: f85f83a9    	ldur	x9, [x29, #-0x8]
100000510: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
100000514: f9400108    	ldr	x8, [x8]
100000518: f9400108    	ldr	x8, [x8]
10000051c: eb090108    	subs	x8, x8, x9
100000520: 54000060    	b.eq	0x10000052c <__Z17test_jagged_arrayv+0x94>
100000524: 14000001    	b	0x100000528 <__Z17test_jagged_arrayv+0x90>
100000528: 9400000d    	bl	0x10000055c <___stack_chk_guard+0x10000055c>
10000052c: b94003e0    	ldr	w0, [sp]
100000530: a9447bfd    	ldp	x29, x30, [sp, #0x40]
100000534: 910143ff    	add	sp, sp, #0x50
100000538: d65f03c0    	ret

000000010000053c <_main>:
10000053c: d10083ff    	sub	sp, sp, #0x20
100000540: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000544: 910043fd    	add	x29, sp, #0x10
100000548: b81fc3bf    	stur	wzr, [x29, #-0x4]
10000054c: 97ffffd3    	bl	0x100000498 <__Z17test_jagged_arrayv>
100000550: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000554: 910083ff    	add	sp, sp, #0x20
100000558: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

000000010000055c <__stubs>:
10000055c: 90000030    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
100000560: f9400610    	ldr	x16, [x16, #0x8]
100000564: d61f0200    	br	x16
