
/Users/jim/work/cppfort/micro-tests/results/memory/mem036-pointer-to-struct/mem036-pointer-to-struct_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

00000001000003b0 <__Z22test_pointer_to_structv>:
1000003b0: d10043ff    	sub	sp, sp, #0x10
1000003b4: 90000008    	adrp	x8, 0x100000000
1000003b8: 91102108    	add	x8, x8, #0x408
1000003bc: f9400109    	ldr	x9, [x8]
1000003c0: 910023e8    	add	x8, sp, #0x8
1000003c4: f90007e9    	str	x9, [sp, #0x8]
1000003c8: f90003e8    	str	x8, [sp]
1000003cc: f94003e8    	ldr	x8, [sp]
1000003d0: b9400108    	ldr	w8, [x8]
1000003d4: f94003e9    	ldr	x9, [sp]
1000003d8: b9400529    	ldr	w9, [x9, #0x4]
1000003dc: 0b090100    	add	w0, w8, w9
1000003e0: 910043ff    	add	sp, sp, #0x10
1000003e4: d65f03c0    	ret

00000001000003e8 <_main>:
1000003e8: d10083ff    	sub	sp, sp, #0x20
1000003ec: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003f0: 910043fd    	add	x29, sp, #0x10
1000003f4: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003f8: 97ffffee    	bl	0x1000003b0 <__Z22test_pointer_to_structv>
1000003fc: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000400: 910083ff    	add	sp, sp, #0x20
100000404: d65f03c0    	ret
