
/Users/jim/work/cppfort/micro-tests/results/memory/mem008-aggregate-init/mem008-aggregate-init_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

00000001000003b0 <__Z19test_aggregate_initv>:
1000003b0: d10043ff    	sub	sp, sp, #0x10
1000003b4: 90000008    	adrp	x8, 0x100000000
1000003b8: 910fe108    	add	x8, x8, #0x3f8
1000003bc: f9400108    	ldr	x8, [x8]
1000003c0: f90007e8    	str	x8, [sp, #0x8]
1000003c4: b9400be8    	ldr	w8, [sp, #0x8]
1000003c8: b9400fe9    	ldr	w9, [sp, #0xc]
1000003cc: 0b090100    	add	w0, w8, w9
1000003d0: 910043ff    	add	sp, sp, #0x10
1000003d4: d65f03c0    	ret

00000001000003d8 <_main>:
1000003d8: d10083ff    	sub	sp, sp, #0x20
1000003dc: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003e0: 910043fd    	add	x29, sp, #0x10
1000003e4: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003e8: 97fffff2    	bl	0x1000003b0 <__Z19test_aggregate_initv>
1000003ec: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003f0: 910083ff    	add	sp, sp, #0x20
1000003f4: d65f03c0    	ret
