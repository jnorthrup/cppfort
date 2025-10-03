
/Users/jim/work/cppfort/micro-tests/results/memory/mem012-global-variable/mem012-global-variable_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

00000001000003f8 <__Z11test_globalv>:
1000003f8: 90000028    	adrp	x8, 0x100004000 <_global_var>
1000003fc: b9400100    	ldr	w0, [x8]
100000400: d65f03c0    	ret

0000000100000404 <_main>:
100000404: d10083ff    	sub	sp, sp, #0x20
100000408: a9017bfd    	stp	x29, x30, [sp, #0x10]
10000040c: 910043fd    	add	x29, sp, #0x10
100000410: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000414: 97fffff9    	bl	0x1000003f8 <__Z11test_globalv>
100000418: a9417bfd    	ldp	x29, x30, [sp, #0x10]
10000041c: 910083ff    	add	sp, sp, #0x20
100000420: d65f03c0    	ret
