
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf010-early-return/cf010-early-return_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z17test_early_returni>:
100000360: 52800c88    	mov	w8, #0x64               ; =100
100000364: 7101901f    	cmp	w0, #0x64
100000368: 1a88b008    	csel	w8, w0, w8, lt
10000036c: 0aa87d00    	bic	w0, w8, w8, asr #31
100000370: d65f03c0    	ret

0000000100000374 <_main>:
100000374: 52800640    	mov	w0, #0x32               ; =50
100000378: d65f03c0    	ret
