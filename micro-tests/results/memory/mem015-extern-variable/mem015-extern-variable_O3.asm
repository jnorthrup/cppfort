
/Users/jim/work/cppfort/micro-tests/results/memory/mem015-extern-variable/mem015-extern-variable_O3.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

00000001000003f8 <__Z11test_externv>:
1000003f8: 90000028    	adrp	x8, 0x100004000 <_extern_var>
1000003fc: b9400100    	ldr	w0, [x8]
100000400: d65f03c0    	ret

0000000100000404 <_main>:
100000404: 90000028    	adrp	x8, 0x100004000 <_extern_var>
100000408: b9400100    	ldr	w0, [x8]
10000040c: d65f03c0    	ret
