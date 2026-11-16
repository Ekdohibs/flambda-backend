(******************************************************************************
 *                                  OxCaml                                    *
 * -------------------------------------------------------------------------- *
 *                               MIT License                                  *
 *                                                                            *
 * Copyright (c) 2026 Jane Street Group LLC                                   *
 * opensource-contacts@janestreet.com                                         *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included    *
 * in all copies or substantial portions of the Software.                     *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER *
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING    *
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER        *
 * DEALINGS IN THE SOFTWARE.                                                  *
 ******************************************************************************)


(* The congruence part of a law of scope, written once.

   A rewrite that goes through a term and changes it in some places
   keeps scope where what it puts in each place keeps the scope of what
   was there. FlambdaScopedProofs.v has that as one lemma for each form
   of term. With the induction of FlambdaTermInductionProofs.v
   ([expr_deep_ind]) the proof for a rewrite is the same few
   applications of those lemmas in every case where the rewrite only
   goes through; the tactic here makes them, and leaves the cases
   where the rewrite does something.

   There is no statement here, and nothing here needs review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaBindersSpec FlambdaScopedProofs FlambdaTermInductionProofs.
Import ListNotations.

(* The handlers of a recursive [Let_cont], each rewritten by a function
   that keeps the parameters and rewrites the body: what
   [expr_deep_ind] knows of the bodies is what is asked. *)
Lemma scoped_handlers_map (f : expr -> expr)
    (g : continuation_handler -> continuation_handler) :
  (∀ params metadata body,
     ∃ metadata',
       g (Handler params metadata body) = Handler params metadata' (f body)) ->
  ∀ handlers : list (continuation * continuation_handler),
    Forall
      (fun kh => scoped (handler_body (snd kh)) (f (handler_body (snd kh))))
      handlers ->
    scoped_handlers handlers (map (fun '(k, h) => (k, g h)) handlers).
Proof.
  intros Hg handlers Hall.
  induction Hall as [ | [k [params metadata body]] handlers Hhead _ IH];
    cbn [map]; [ apply scoped_handlers_nil | ].
  destruct (Hg params metadata body) as (metadata' & ->).
  apply scoped_handlers_cons; [ apply scoped_handler_intro, Hhead | exact IH ].
Qed.

Lemma map_fst_rewritten {A B : Type} (g : B -> B) (l : list (A * B)) :
  map fst (map (fun '(k, h) => (k, g h)) l) = map fst l.
Proof. rewrite map_map. apply map_ext. intros [k h]. reflexivity. Qed.

(* The two forms of term that hold applications of continuations. *)
Lemma scoped_apply_cont (ac ac' : apply_cont) :
  incl (free_variables_of_apply_cont ac') (free_variables_of_apply_cont ac) ->
  incl (apply_cont_continuations ac') (apply_cont_continuations ac) ->
  scoped (Apply_cont ac) (Apply_cont ac').
Proof. intros Hv Hk _. scope. split; [ reflexivity | split; assumption ]. Qed.

Lemma scoped_switch_map (g : apply_cont -> apply_cont) (scrutinee : simple)
    (arms : list (Z * apply_cont)) :
  (∀ ac,
     incl (free_variables_of_apply_cont (g ac))
       (free_variables_of_apply_cont ac)) ->
  (∀ ac,
     incl (apply_cont_continuations (g ac)) (apply_cont_continuations ac)) ->
  scoped (Switch scrutinee arms)
    (Switch scrutinee (map (fun '(n, ac) => (n, g ac)) arms)).
Proof.
  intros Hv Hk _. scope. split; [ reflexivity | split ].
  - apply incl_app_app; [ apply incl_refl | ].
    induction arms as [ | [n ac] arms IH]; [ apply incl_refl | ].
    cbn [map flat_map snd]. apply incl_app_app; [ apply Hv | exact IH ].
  - induction arms as [ | [n ac] arms IH]; [ apply incl_refl | ].
    cbn [map flat_map snd]. apply incl_app_app; [ apply Hk | exact IH ].
Qed.

(* One step: a hypothesis of the induction, a term that is not changed,
   or a form of term that the rewrite goes through. *)
Ltac scoped_step :=
  first
    [ assumption
    | apply scoped_refl
    | apply scoped_let; [ apply incl_refl | ]
    | apply scoped_let_static; [ apply scoped_bindings_refl | ]
    | apply scoped_let_cont
    | apply scoped_handler_intro
    | apply scoped_let_cont_recursive;
      [ first [ reflexivity | apply map_fst_rewritten ]
      | first
          [ apply scoped_handlers_refl
          | eapply scoped_handlers_map; [ | eassumption ];
            intros; eexists; reflexivity ]
      | ] ].

(* The goal is [scoped e (f e)] in a case of [expr_deep_ind], with [f]
   reduced one step: what is left are the cases where [f] does
   something. *)
Ltac scoped_congruence := repeat scoped_step.
