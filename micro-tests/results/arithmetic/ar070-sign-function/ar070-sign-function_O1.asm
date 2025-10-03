
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar070-sign-function/ar070-sign-function_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z9test_signi>:
100000360: 131f7c08    	asr	w8, w0, #31
100000364: 7100041f    	cmp	w0, #0x1
100000368: 1a9fb500    	csinc	w0, w8, wzr, lt
10000036c: d65f03c0    	ret

0000000100000370 <_main>:
100000370: 12800000    	mov	w0, #-0x1               ; =-1
100000374: d65f03c0    	ret
