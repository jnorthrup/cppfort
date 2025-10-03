
/Users/jim/work/cppfort/micro-tests/results/memory/mem011-string-literal/mem011-string-literal_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

00000001000003b0 <__Z19test_string_literalv>:
1000003b0: 90000000    	adrp	x0, 0x100000000
1000003b4: 910f8000    	add	x0, x0, #0x3e0
1000003b8: d65f03c0    	ret

00000001000003bc <_main>:
1000003bc: d10083ff    	sub	sp, sp, #0x20
1000003c0: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003c4: 910043fd    	add	x29, sp, #0x10
1000003c8: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003cc: 97fffff9    	bl	0x1000003b0 <__Z19test_string_literalv>
1000003d0: 39c00000    	ldrsb	w0, [x0]
1000003d4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003d8: 910083ff    	add	sp, sp, #0x20
1000003dc: d65f03c0    	ret
