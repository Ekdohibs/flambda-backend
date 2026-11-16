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

(* Proofs of the statements of FlambdaMeetTermsSpec.v. Nothing here
   needs review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar FlambdaBindingTime
  FlambdaTypingEnv FlambdaTypingEnvSpec FlambdaMeetGrammar FlambdaExpandHead
  FlambdaMeetEnv FlambdaMeet FlambdaMeetSpec FlambdaMeetStepsSpec
  FlambdaMeetStepsProofs FlambdaMeetEnvSpec FlambdaMeetEnvProofs
  FlambdaMeetScopesSpec FlambdaMeetScopesProofs FlambdaMeetImmediateSpec
  FlambdaMeetImmediateProofs FlambdaMeetFuelSpec FlambdaMeetFuelProofs
  FlambdaMeetClosuresSpec FlambdaMeetClosuresProofs FlambdaMeetKeepsSpec
  FlambdaMeetKeepsProofs FlambdaMeetTermsSpec.
Import ListNotations.

(* The meet of types of a fuel at which the meet of expanded types
   has what it is to keep: a variable that the environment knew has
   the name mode that it had, and one that it did not know may not
   occur in terms. *)
Lemma meet_of_fuel_keeps_terms (join : join_scoped_envs)
    (choices : meet_choices) (fuel : nat) :
  meet_expanded_head_keeps (Meet.meet_expanded_head join choices fuel) ->
  meet_keeps_terms (Meet.meet join choices fuel).
Proof.
  intros Hkeeps tenv tenv' t1 t2 t (Hfits & Ht1 & Ht2 & Hkinds) Hmeet x Hx.
  set (mh := Meet.meet_expanded_head join choices fuel) in *.
  unfold Meet.meet in Hmeet. fold mh in Hmeet.
  destruct (Typing_env.is_bottom tenv); [ discriminate Hmeet | ].
  destruct (Meet_env.meet_type (Meet_env.create tenv) t1 t2 mh)
    as [ [ r0 | r menv' ] | ] eqn:Htype; try discriminate Hmeet.
  cbv beta iota zeta in Hmeet.
  destruct (is_obviously_bottom (extract_value r t1 t2));
    [ discriminate Hmeet | ].
  cbn in Hmeet. injection Hmeet as <- <-.
  destruct (meet_type_keeps mh (Meet_env.create tenv) menv' t1 t2 r Hkeeps
              Hfits Ht1 Ht2 Htype)
    as ([[Hextends _] Hnames] & _ & _).
  unfold in_scope_in_terms in *.
  destruct (Typing_env.mem None tenv (Name_var x)) eqn:Hknown.
  - exact (eq_trans
             (eq_sym
                (meet_env_extends_mode (Meet_env.create tenv) menv'
                   (Some Name_mode.normal) x Hextends Hknown))
             Hx).
  - exfalso.
    assert (Hknown' :
      Typing_env.mem None (Meet_env.typing_env menv') (Name_var x) = true).
    { destruct (Typing_env.mem None (Meet_env.typing_env menv') (Name_var x))
        eqn:Hmem; [ reflexivity | ].
      rewrite (unknown_not_in_terms _ x Hmem) in Hx. discriminate Hx. }
    destruct (Hnames x Hknown' Hknown) as (_ & _ & Hterms).
    rewrite Hterms in Hx. discriminate Hx.
Qed.

Theorem meet_to_prove_keeps_terms : meet_to_prove_keeps_terms_statement.
Proof.
  intros join fuel Hjoin. unfold meet_to_prove.
  destruct (meet_expanded_head_fuel_ok join repaired_choices eq_refl
              (fun mh Hkeeps Hsound _ =>
                 value_heads_of_join join mh Hjoin Hkeeps Hsound)
              fuel) as [Hkeeps _].
  exact (meet_of_fuel_keeps_terms join repaired_choices fuel Hkeeps).
Qed.

Theorem meet_to_prove_keeps_terms_of_keeps
  : meet_to_prove_keeps_terms_of_keeps_statement.
Proof.
  intros join fuel Hjoin. unfold meet_to_prove.
  exact (meet_of_fuel_keeps_terms join repaired_choices fuel
           (meet_expanded_head_repaired_keeps join fuel Hjoin)).
Qed.
