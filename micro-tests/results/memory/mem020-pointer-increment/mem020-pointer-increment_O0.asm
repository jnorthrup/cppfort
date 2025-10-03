
/Users/jim/work/cppfort/micro-tests/results/memory/mem020-pointer-increment/mem020-pointer-increment_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000498 <__Z22test_pointer_incrementv>:
100000498: d10103ff    	sub	sp, sp, #0x40
10000049c: a9037bfd    	stp	x29, x30, [sp, #0x30]
1000004a0: 9100c3fd    	add	x29, sp, #0x30
1000004a4: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
1000004a8: f9400108    	ldr	x8, [x8]
1000004ac: f9400108    	ldr	x8, [x8]
1000004b0: f81f83a8    	stur	x8, [x29, #-0x8]
1000004b4: 90000009    	adrp	x9, 0x100000000 <___stack_chk_guard+0x100000000>
1000004b8: 91151129    	add	x9, x9, #0x544
1000004bc: 3dc00120    	ldr	q0, [x9]
1000004c0: 910043e8    	add	x8, sp, #0x10
1000004c4: 3d8007e0    	str	q0, [sp, #0x10]
1000004c8: b9401129    	ldr	w9, [x9, #0x10]
1000004cc: b90023e9    	str	w9, [sp, #0x20]
1000004d0: f90007e8    	str	x8, [sp, #0x8]
1000004d4: f94007e8    	ldr	x8, [sp, #0x8]
1000004d8: 91001109    	add	x9, x8, #0x4
1000004dc: f90007e9    	str	x9, [sp, #0x8]
1000004e0: b9400508    	ldr	w8, [x8, #0x4]
1000004e4: b90007e8    	str	w8, [sp, #0x4]
1000004e8: f85f83a9    	ldur	x9, [x29, #-0x8]
1000004ec: 90000028    	adrp	x8, 0x100004000 <___stack_chk_guard+0x100004000>
1000004f0: f9400108    	ldr	x8, [x8]
1000004f4: f9400108    	ldr	x8, [x8]
1000004f8: eb090108    	subs	x8, x8, x9
1000004fc: 54000060    	b.eq	0x100000508 <__Z22test_pointer_incrementv+0x70>
100000500: 14000001    	b	0x100000504 <__Z22test_pointer_incrementv+0x6c>
100000504: 9400000d    	bl	0x100000538 <___stack_chk_guard+0x100000538>
100000508: b94007e0    	ldr	w0, [sp, #0x4]
10000050c: a9437bfd    	ldp	x29, x30, [sp, #0x30]
100000510: 910103ff    	add	sp, sp, #0x40
100000514: d65f03c0    	ret

0000000100000518 <_main>:
100000518: d10083ff    	sub	sp, sp, #0x20
10000051c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000520: 910043fd    	add	x29, sp, #0x10
100000524: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000528: 97ffffdc    	bl	0x100000498 <__Z22test_pointer_incrementv>
10000052c: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000530: 910083ff    	add	sp, sp, #0x20
100000534: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

0000000100000538 <__stubs>:
100000538: 90000030    	adrp	x16, 0x100004000 <___stack_chk_guard+0x100004000>
10000053c: f9400610    	ldr	x16, [x16, #0x8]
100000540: d61f0200    	br	x16
