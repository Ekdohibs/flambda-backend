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

(* Lemmas for the proofs of the statements of FlambdaRegionsEndsSpec.v:
   what is said of a continuation ([cont_ends], [ret_ends],
   [raise_ends]) is kept where the frames that were drawn before the
   code of its user was entered are kept, and for a later user of the
   same code. Nothing here is about a step of the machine, and nothing
   here needs review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec FlambdaSemanticsSpec
  FlambdaRegionsSpec FlambdaRegionsCallsSpec FlambdaRegionsCheckSpec
  FlambdaRegionsCheckedSpec.
From OxCamlFlambda2 Require Import FlambdaRegionsEndsSpec
  FlambdaRegionsEndsBasicsProofs.
Import ListNotations.

Scheme cont_ends_mut := Induction for cont_ends Sort Prop
  with ret_ends_mut := Induction for ret_ends Sort Prop
  with raise_ends_mut := Induction for raise_ends Sort Prop.
Combined Scheme conts_ends_mutind
  from cont_ends_mut, ret_ends_mut, raise_ends_mut.

(* ======================================================================== *)
(* Other frames, the same before the user's code was entered                *)
(* ======================================================================== *)

Section Kept_conts.

Variable cert : certificate.
Variable opts : ends_options.
Variable ret : continuation.
Variable bottom : symbol.
Variables rs rs' : region_state.

Lemma kept_called_from (u : ends_user) (a : ends_caller) :
  frames_before (eu_floor u) (rs_frames rs')
  = frames_before (eu_floor u) (rs_frames rs) ->
  ends_called_from rs u a ->
  ends_called_from rs' u a.
Proof.
  intros Hkept (Hplaced & Hbelow & Hfloor & Htight).
  assert (Hlt : ∀ m : mark, In m (ec_marks a) -> (m < eu_floor u)%positive).
  { intros m Hin. apply Pos.ltb_lt.
    exact (proj1 (forallb_forall _ _) Hbelow m Hin). }
  split;
    [ exact
        (kept_ends_placed _ _ _ Hkept (ec_vars a) (ec_state a) (ec_marks a)
           (ec_floor a) Hlt Hplaced)
    | ].
  split; [ exact Hbelow | ]. split; [ exact Hfloor | ].
  apply (kept_tight _ _ _ Hkept); [ | exact Htight ].
  destruct (stale_stop (es_stale (ec_state a)) (ec_floor a) (ec_marks a));
    lia.
Qed.

Lemma kept_conts :
  (∀ (u : ends_user) (ec : ends_cont) (c : cont_value),
     cont_ends cert opts ret bottom rs u ec c ->
     frames_before (eu_floor u) (rs_frames rs')
     = frames_before (eu_floor u) (rs_frames rs) ->
     cont_ends cert opts ret bottom rs' u ec c)
  ∧ (∀ (a : ends_caller) (c : cont_value),
       ret_ends cert opts ret bottom rs a c ->
       frames_before (ec_floor a) (rs_frames rs')
       = frames_before (ec_floor a) (rs_frames rs) ->
       ret_ends cert opts ret bottom rs' a c)
  ∧ (∀ (a : ends_caller) (c : cont_value),
       raise_ends cert opts ret bottom rs a c ->
       frames_before (ec_floor a) (rs_frames rs')
       = frames_before (ec_floor a) (rs_frames rs) ->
       raise_ends cert opts ret bottom rs' a c).
Proof.
  apply (conts_ends_mutind cert opts ret bottom rs
           (fun u ec c _ =>
              frames_before (eu_floor u) (rs_frames rs')
              = frames_before (eu_floor u) (rs_frames rs) ->
              cont_ends cert opts ret bottom rs' u ec c)
           (fun a c _ =>
              frames_before (ec_floor a) (rs_frames rs')
              = frames_before (ec_floor a) (rs_frames rs) ->
              ret_ends cert opts ret bottom rs' a c)
           (fun a c _ =>
              frames_before (ec_floor a) (rs_frames rs')
              = frames_before (ec_floor a) (rs_frames rs) ->
              raise_ends cert opts ret bottom rs' a c)).
  - intros u params metadata handler entry vars conts recursive es base
      Hmade Hexpr Hdistinct Hnew Hopen Hin Hconts IH Hnone Hkept.
    apply (Ends_ok_handler cert opts ret bottom rs' u params metadata handler
             entry vars conts recursive es base);
      try assumption.
    intros k ec Hk. exact (IH k ec Hk Hkept).
  - intros u c _. apply Ends_ok_unreachable.
  - intros u Hfloor _. apply Ends_ok_exit. exact Hfloor.
  - intros u a c Hbottom Hcalled Hret IH Hkept.
    apply (Ends_ok_return cert opts ret bottom rs' u a c Hbottom);
      [ exact (kept_called_from u a Hkept Hcalled) | ].
    apply IH. destruct Hcalled as (_ & _ & Hfloor & _).
    apply Pos.leb_le in Hfloor.
    exact (frames_before_le _ _ _ _ Hfloor Hkept).
  - intros u k extra Hk _. apply Ends_ok_raise_out. exact Hk.
  - intros u a c Hbottom Hcalled Hraise IH Hkept.
    apply (Ends_ok_raise cert opts ret bottom rs' u a c Hbottom);
      [ exact (kept_called_from u a Hkept Hcalled) | ].
    apply IH. destruct Hcalled as (_ & _ & Hfloor & _).
    apply Pos.leb_le in Hfloor.
    exact (frames_before_le _ _ _ _ Hfloor Hkept).
  - intros a k c Hjump Hsome IH Hkept.
    apply (Ends_ret_to cert opts ret bottom rs' a k c Hjump).
    intros ec Hk. exact (IH ec Hk Hkept).
  - intros a remaining remaining_arity return_arity return_to exn_to
      Hret IHret Hraise IHraise Hkept.
    apply Ends_ret_over; [ exact (IHret Hkept) | exact (IHraise Hkept) ].
  - intros a k simples extra c Heval Hraise Hsome IH Hkept.
    apply (Ends_raise_to cert opts ret bottom rs' a k simples extra c Heval
             Hraise).
    intros ec Hk. exact (IH ec Hk Hkept).
Qed.

Lemma kept_cont_ends (u : ends_user) (ec : ends_cont) (c : cont_value) :
  frames_before (eu_floor u) (rs_frames rs')
  = frames_before (eu_floor u) (rs_frames rs) ->
  cont_ends cert opts ret bottom rs u ec c ->
  cont_ends cert opts ret bottom rs' u ec c.
Proof. intros Hkept Hok. exact (proj1 kept_conts u ec c Hok Hkept). Qed.

Lemma kept_ret_ends (a : ends_caller) (c : cont_value) :
  frames_before (ec_floor a) (rs_frames rs')
  = frames_before (ec_floor a) (rs_frames rs) ->
  ret_ends cert opts ret bottom rs a c ->
  ret_ends cert opts ret bottom rs' a c.
Proof.
  intros Hkept Hok. exact (proj1 (proj2 kept_conts) a c Hok Hkept).
Qed.

Lemma kept_raise_ends (a : ends_caller) (c : cont_value) :
  frames_before (ec_floor a) (rs_frames rs')
  = frames_before (ec_floor a) (rs_frames rs) ->
  raise_ends cert opts ret bottom rs a c ->
  raise_ends cert opts ret bottom rs' a c.
Proof.
  intros Hkept Hok. exact (proj2 (proj2 kept_conts) a c Hok Hkept).
Qed.

Lemma kept_conts_ends (u : ends_user)
    (entries : list (continuation * ends_cont)) (conts : cont_env) :
  frames_before (eu_floor u) (rs_frames rs')
  = frames_before (eu_floor u) (rs_frames rs) ->
  conts_ends cert opts ret bottom rs u entries conts ->
  conts_ends cert opts ret bottom rs' u entries conts.
Proof.
  intros Hkept [Hsome Hnone]. split; [ | exact Hnone ].
  intros k ec Hk. exact (kept_cont_ends u ec _ Hkept (Hsome k ec Hk)).
Qed.

End Kept_conts.

(* ======================================================================== *)
(* A later user of the same code                                            *)
(* ======================================================================== *)

(* The variables of regions that the first user has are bound as they
   were, and are variables of regions still; the regions of a [try]
   are still known to be; the code is the same. *)
Definition ends_user_extends (u u' : ends_user) : Prop :=
  (∀ r : variable,
     among r (eu_regions u) = true ->
     lookup r (eu_vars u') = lookup r (eu_vars u)
     ∧ among r (eu_regions u') = true)
  ∧ (∀ r : variable,
       among r (eu_tries u) = true -> among r (eu_tries u') = true)
  ∧ eu_floor u' = eu_floor u.

Lemma user_cont_ends (cert : certificate) (opts : ends_options)
    (ret : continuation) (bottom : symbol) (rs : region_state)
    (u u' : ends_user) (ec : ends_cont) (c : cont_value) :
  ends_user_extends u u' ->
  cont_ends cert opts ret bottom rs u ec c ->
  cont_ends cert opts ret bottom rs u' ec c.
Proof.
  intros (Hregions & Htries & Hfloor) Hok.
  destruct Hok
    as [u params metadata handler entry vars conts recursive es base
          Hmade Hexpr Hdistinct Hnew Hopen Hin Hconts Hnone
       | u c | u Hbot | u a c Hbot Hcalled Hret | u k extra Hk
       | u a c Hbot Hcalled Hraise].
  - apply (Ends_ok_handler cert opts ret bottom rs u' params metadata handler
             entry vars conts recursive es base);
      try assumption.
    + destruct Hmade as (H1 & H2 & H3 & H4 & H5 & H6).
      repeat (split; [ assumption | ]). split.
      * intros r Hr. destruct (H5 r Hr) as [Hlookup Hamong].
        destruct (Hregions r Hamong) as [Hlookup' Hamong'].
        split; [ rewrite Hlookup'; exact Hlookup | exact Hamong' ].
      * intros r Hr. exact (Htries r (H6 r Hr)).
    + rewrite Hfloor. exact Hconts.
  - apply Ends_ok_unreachable.
  - apply Ends_ok_exit. rewrite Hfloor. exact Hbot.
  - apply (Ends_ok_return cert opts ret bottom rs u' a c Hbot);
      [ | exact Hret ].
    unfold ends_called_from in *. rewrite Hfloor. exact Hcalled.
  - apply Ends_ok_raise_out. exact Hk.
  - apply (Ends_ok_raise cert opts ret bottom rs u' a c Hbot);
      [ | exact Hraise ].
    unfold ends_called_from in *. rewrite Hfloor. exact Hcalled.
Qed.

Lemma user_conts_ends (cert : certificate) (opts : ends_options)
    (ret : continuation) (bottom : symbol) (rs : region_state)
    (u u' : ends_user) (entries : list (continuation * ends_cont))
    (conts : cont_env) :
  ends_user_extends u u' ->
  conts_ends cert opts ret bottom rs u entries conts ->
  conts_ends cert opts ret bottom rs u' entries conts.
Proof.
  intros Hextends [Hsome Hnone]. split; [ | exact Hnone ].
  intros k ec Hk.
  exact
    (user_cont_ends cert opts ret bottom rs u u' ec _ Hextends
       (Hsome k ec Hk)).
Qed.
