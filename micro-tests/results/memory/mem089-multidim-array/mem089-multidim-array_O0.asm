
/Users/jim/work/cppfort/micro-tests/results/memory/mem089-multidim-array/mem089-multidim-array_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000498 <__Z19test_multidim_arrayv>:
100000498: d10103ff    	sub	sp, sp, #0x40
10000049c: a9037bfd    	stp	x29, x30, [sp, #0x30]
1000004a0: 9100c3fd    	add	x29, sp, #0x30
1000004a4: 90000028    	adrp	x8, 0x100004000 <_memcpy+0x100004000>
1000004a8: f9400108    	ldr	x8, [x8]
1000004ac: f9400108    	ldr	x8, [x8]
1000004b0: f81f83a8    	stur	x8, [x29, #-0x8]
1000004b4: 910013e0    	add	x0, sp, #0x4
1000004b8: d2800482    	mov	x2, #0x24               ; =36
1000004bc: 90000001    	adrp	x1, 0x100000000 <_memcpy+0x100000000>
1000004c0: 9114e021    	add	x1, x1, #0x538
1000004c4: 94000017    	bl	0x100000520 <_memcpy+0x100000520>
1000004c8: b94017e8    	ldr	w8, [sp, #0x14]
1000004cc: b90003e8    	str	w8, [sp]
1000004d0: f85f83a9    	ldur	x9, [x29, #-0x8]
1000004d4: 90000028    	adrp	x8, 0x100004000 <_memcpy+0x100004000>
1000004d8: f9400108    	ldr	x8, [x8]
1000004dc: f9400108    	ldr	x8, [x8]
1000004e0: eb090108    	subs	x8, x8, x9
1000004e4: 54000060    	b.eq	0x1000004f0 <__Z19test_multidim_arrayv+0x58>
1000004e8: 14000001    	b	0x1000004ec <__Z19test_multidim_arrayv+0x54>
1000004ec: 94000010    	bl	0x10000052c <_memcpy+0x10000052c>
1000004f0: b94003e0    	ldr	w0, [sp]
1000004f4: a9437bfd    	ldp	x29, x30, [sp, #0x30]
1000004f8: 910103ff    	add	sp, sp, #0x40
1000004fc: d65f03c0    	ret

0000000100000500 <_main>:
100000500: d10083ff    	sub	sp, sp, #0x20
100000504: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000508: 910043fd    	add	x29, sp, #0x10
10000050c: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000510: 97ffffe2    	bl	0x100000498 <__Z19test_multidim_arrayv>
100000514: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000518: 910083ff    	add	sp, sp, #0x20
10000051c: d65f03c0    	ret

Disassembly of section __TEXT,__stubs:

0000000100000520 <__stubs>:
100000520: 90000030    	adrp	x16, 0x100004000 <_memcpy+0x100004000>
100000524: f9400610    	ldr	x16, [x16, #0x8]
100000528: d61f0200    	br	x16
10000052c: 90000030    	adrp	x16, 0x100004000 <_memcpy+0x100004000>
100000530: f9400a10    	ldr	x16, [x16, #0x10]
100000534: d61f0200    	br	x16
